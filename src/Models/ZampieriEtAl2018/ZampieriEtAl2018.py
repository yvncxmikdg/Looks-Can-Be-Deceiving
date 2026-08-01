import torch
from torch import nn
import torch.nn.functional as F

from modeling.Models.Maskable import Maskable
from modeling.Models.MaskedUNet.alignment_parts import AlignmentUp
from modeling.Models.ZampieriEtAl2018.vf_map import diffeomorphic_composition_via_sequential_flows, warp_batched_polygons_according_to_vector_fields
from modeling.Models.ModelDatum import ModelOutput, CHANNEL_INPUT, BUILDINGS, DISPLACEMENT_FIELD, DO_SOFTMAX
from modeling.utils.sample_generator_utils import draw_objects_on_mask


class ZampieriEtAl2018(Maskable, nn.Module):
    def __init__(self,
                 n_classes,
                 input_channel_mask_index=-1,
                 output_channel_background_index=-1,
                 chain_steps=4,
                 initial_x_dim=256,
                 initial_y_dim=256,
                 mask_channel=3,
                 chain_combination_strategy="composition"):
        Maskable.__init__(self, n_classes, input_channel_mask_index, output_channel_background_index)
        nn.Module.__init__(self)
        self.n_classes = n_classes
        self.chain_steps = chain_steps
        self.initial_x_dim = initial_x_dim
        self.initial_y_dim = initial_y_dim
        self.mask_channel = mask_channel
        self.chain_combination_strategy = chain_combination_strategy

        self._d_s = nn.ModuleList([self._build_block() for _ in range(0, self.chain_steps)])

    def _build_block(self):
        # Subclasses override this to swap in an alternative block (e.g. attention-augmented).
        return ZampieriEtAl2018_block(self.mask_channel)

    def forward(self, model_input):

        #Pull the imagery (rgb + building-footprint mask) and the building polygons from the ModelInput
        imagery = model_input[CHANNEL_INPUT]
        batched_buildings = model_input[BUILDINGS]
        do_softmax = model_input[DO_SOFTMAX]
        batch_size = imagery.shape[0]

        batched_polygons = []
        for buildings in batched_buildings:
            batched_polygons.append([b.getGeometry("pixels") for b in buildings])

        # Store the shape so we can refer to it later
        batch_size, _, source_x_dim, source_y_dim = imagery.shape

        # Create lists for storing the vector fields we will be generating, and their respective ODE functions
        intermediate_dispacement_maps = []
        composed_vector_field = torch.zeros((imagery.shape[0], 2, self.initial_x_dim, self.initial_y_dim)).to(imagery.get_device())

        # Run each block at its respective scale
        for s, block in enumerate(self._d_s):

            # Compute the dimensions of the step we are on currently.
            scale = (self.chain_steps - s) - 1
            step_x_dim = self.initial_x_dim//(2**scale)
            step_y_dim = self.initial_y_dim//(2**scale)

            # Downsample the image to the dimension we are working with
            x_visual = F.interpolate(imagery[:, :self.mask_channel, :, :], [step_x_dim, step_y_dim])

            # If we have generated displacement maps of any kind...
            if len(intermediate_dispacement_maps) > 0:

                # Warp the polygons according to the displacement maps
                batched_polygons = warp_batched_polygons_according_to_vector_fields(batched_polygons,
                                                                                    intermediate_dispacement_maps[-1],
                                                                                    source_x_dim,
                                                                                    source_y_dim)

                # Redraw the building polygons based on the new sizes
                x_cadastral_warped_list = []
                for i in range(0, batch_size):
                    x_cadastral_i = draw_objects_on_mask(batched_polygons[i],
                                                         0,
                                                         0,
                                                         imagery.shape[2],
                                                         imagery.shape[3],
                                                         step_x_dim,
                                                         step_y_dim,
                                                         channels=1,
                                                         geometry_accessor=lambda x: x)
                    x_cadastral_warped_list.append(torch.tensor(x_cadastral_i).to(imagery.get_device()))

                # Warp the cadastral image according to the displacement map
                x_cadastral_warped = torch.stack(x_cadastral_warped_list, dim=0).unsqueeze(1)

            #If we havent generated any displacement maps, then just use the inital data
            else:
                x_cadastral_warped = F.interpolate(imagery[:, self.mask_channel, :, :].unsqueeze(1), [step_x_dim, step_y_dim])

            # Combine the data back into a tensor we can pass to the module.
            input_data = torch.cat([x_visual, x_cadastral_warped], dim=1)

            # Run the module.
            batched_displacement_map = block(input_data)

            # Upscale the generated displacement so that it is the right dimension and we can add it together with the existing map
            upscaled_batched_displacement_map = F.interpolate(batched_displacement_map, [self.initial_x_dim, self.initial_y_dim])

            # Compose all the vector fields that have been generated up to this point
            if self.chain_combination_strategy == "composition":
                composed_vector_field = diffeomorphic_composition_via_sequential_flows([composed_vector_field, upscaled_batched_displacement_map])
            elif self.chain_combination_strategy == "addition":
                composed_vector_field = composed_vector_field + upscaled_batched_displacement_map
            else:
                raise ValueError("Unknown combination strategy \"" + self.chain_combination_strategy + "\" options are compsition and addition.")

            # Update the lists with the most recent displacement map, and corresponding ODE function
            intermediate_dispacement_maps.append(upscaled_batched_displacement_map)

        result = ModelOutput()
        result.setField(DISPLACEMENT_FIELD, composed_vector_field)
        result.setField(DO_SOFTMAX, do_softmax)
        return result

    def load(self, path, strict=True):
        checkpoint = torch.load(path, map_location="cpu")
        new_state_dict = {key.replace("model.", "", 1): v for key, v in checkpoint["state_dict"].items()}

        # Some blocks (e.g. the attention-augmented variant) create parameters lazily on the first
        # forward pass, so they don't exist on a freshly-constructed model. Pre-register any such
        # missing keys at the checkpoint's shapes so the trained values load instead of being
        # re-randomized. For the plain block every key already exists, so this loop is a no-op.
        existing = set(self.state_dict().keys())
        for key, value in new_state_dict.items():
            if key in existing:
                continue
            *parent_path, attr = key.split(".")
            module = self
            for part in parent_path:
                module = module[int(part)] if part.isdigit() else getattr(module, part)
            # setattr (not register_parameter) so the lazily-set None placeholder is replaced.
            setattr(module, attr, nn.Parameter(torch.zeros(value.shape, dtype=torch.float32)))

        self.load_state_dict(new_state_dict, strict=strict)

#Architecture is from "Multimodal image alignment through a multiscale chain of neural networks with application to remote sensing"
#https://openaccess.thecvf.com/content_ECCV_2018/papers/Armand_Zampieri_Multimodal_image_alignment_ECCV_2018_paper.pdf
#AlignmentUp module added after the fact because the paper does not provide sufficent detail as to how their deconvolutional layer was implemented
class ZampieriEtAl2018_block(nn.Module):
    def __init__(self, mask_channel=3):
        super().__init__()

        self._mask_channel = mask_channel

        self.visual_layer_block_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2, bias=True, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, bias=True, padding_mode="reflect"),
            nn.LeakyReLU())

        self.visual_layer_block_2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.LeakyReLU())

        self.visual_layer_block_3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.LeakyReLU())

        self.cadastral_layer_block_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2, bias=True, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, bias=True, padding_mode="reflect"),
            nn.LeakyReLU())

        self.cadastral_layer_block_2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.LeakyReLU())

        self.cadastral_layer_block_3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.LeakyReLU())

        self.down_1_combo_layer_block_start = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.LeakyReLU())

        self.down_2_combo_layer_block_start = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.LeakyReLU())

        self.down_3_combo_layer_block = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.LeakyReLU())
        self.down_3_combo_layer_up = AlignmentUp(64)

        self.down_2_combo_layer_block_end = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.LeakyReLU())
        self.down_2_combo_layer_up = AlignmentUp(32)

        self.down_1_combo_layer_block_end = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"))

    # Attention-augmented subclasses inject attention by overriding these hooks; the plain block
    # leaves them as identities so its behavior is unchanged.
    def _attend_visual(self, a6, b6):  # pylint: disable=unused-argument
        return a6

    def _attend_bottleneck(self, down_3_output):
        return down_3_output

    def _attend_decoder(self, down_2_complete_intermediate):
        return down_2_complete_intermediate

    def forward(self, x):
        #Notation based on architecture diagram (figure 4) on page 6
        #Hyperparamters based on contents of hyperparamter table (figure 5) on page 6
        #https://www.lri.fr/~gcharpia/alignment/supp_mat_alignment.pdf
        x_visual = x[:, :self._mask_channel, :, :].mean(dim=1).unsqueeze(1)
        x_cadastral = x[:, self._mask_channel, :, :].unsqueeze(1)

        a3 = self.visual_layer_block_1(x_visual)
        b3 = self.cadastral_layer_block_1(x_cadastral)

        a6 = self.visual_layer_block_2(a3)
        b6 = self.cadastral_layer_block_2(b3)
        a6 = self._attend_visual(a6, b6)

        a9 = self.visual_layer_block_3(a6)
        b9 = self.cadastral_layer_block_3(b6)

        down_1_input = torch.cat([a3, b3], dim=1)
        down_1_intermediate = self.down_1_combo_layer_block_start(down_1_input)

        down_2_input = torch.cat([a6, b6], dim=1)
        down_2_intermediate = self.down_2_combo_layer_block_start(down_2_input)

        down_3_input = torch.cat([a9, b9], dim=1)
        down_3_output = self.down_3_combo_layer_block(down_3_input)
        down_3_output = self._attend_bottleneck(down_3_output)

        down_2_complete_intermediate = self.down_3_combo_layer_up(down_3_output, down_2_intermediate)
        down_2_complete_intermediate = self._attend_decoder(down_2_complete_intermediate)
        down_2_output = self.down_2_combo_layer_block_end(down_2_complete_intermediate)

        down_1_complete_intermediate = self.down_2_combo_layer_up(down_2_output, down_1_intermediate)
        output = self.down_1_combo_layer_block_end(down_1_complete_intermediate)

        return output
