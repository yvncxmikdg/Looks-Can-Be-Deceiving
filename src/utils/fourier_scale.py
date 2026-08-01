# pylint: disable=not-callable
import torch
import torch.nn.functional as F

def strip_and_pad(source_matrix, internal_x_max, internal_y_max, projection_mode="magnitude"):
    #Make sure that the passed projection mode is valid...
    PROJECTION_MODES = ["magnitude", "visual", "none"]
    if projection_mode.lower() not in PROJECTION_MODES:
        raise ValueError("Invalid projection mode \"" + str(projection_mode) + "\" valid options are " + str(PROJECTION_MODES))

    #First we need to compute how much space in the spectrogram is going to be added or removed along each axis
    #This will give us the factor by which we need to increase the amplitude of the remaining elements in the spectrogram
    x_reduction_factor = internal_x_max/source_matrix.shape[0]
    y_reduction_factor = internal_y_max/source_matrix.shape[1]
    spectrogram_scale_factor = x_reduction_factor*y_reduction_factor

    #Check if we are dealing with a WxH or WxHxC matrix
    has_channels = len(source_matrix.shape) == 3

    #Now compute how much we need to trim off of this matrix in order to fit it into the fixed representation space
    left_trim_start = max(0, int(source_matrix.shape[0]/2 - (internal_x_max/2)))
    right_trim_end  = max(0, int(source_matrix.shape[0]/2 + (internal_x_max/2)))
    top_trim_start  = max(0, int(source_matrix.shape[1]/2 - (internal_y_max/2)))
    bottom_trim_end = max(0, int(source_matrix.shape[1]/2 + (internal_y_max/2)))

    #Trim the matrix by the amounts we computed, and optionally include the final channel depending on if we have it or not
    if has_channels:
        result = source_matrix[left_trim_start:right_trim_end, top_trim_start:bottom_trim_end, :]
    else:
        result = source_matrix[left_trim_start:right_trim_end, top_trim_start:bottom_trim_end]

    #Next we need to do some math to determine how much we need to pad the matrix, we need to pad on the left and right side
    #of the center because that is where the low frequency data is. Also we need to do some math to handle the case if there
    #is odd data.
    x_dim, y_dim = result.shape[:2]
    x_odd = ((internal_x_max-x_dim)%2) == 1
    y_odd = ((internal_y_max-y_dim)%2) == 1
    xy_pad_shape = [(internal_y_max-y_dim)//2 + y_odd,
                    (internal_y_max-y_dim)//2,
                    (internal_x_max-x_dim)//2 + x_odd,
                    (internal_x_max-x_dim)//2
                   ]
    pad_shape = [0, 0] + xy_pad_shape if has_channels else xy_pad_shape

    #Actually pad the matrix
    result = F.pad(result, pad=pad_shape)

    #Optionally, multiply the content of the matrix by a scale factor
    if projection_mode == "visual" or (projection_mode == "magnitude" and spectrogram_scale_factor > 1):
        result *= spectrogram_scale_factor

    #Return our result
    return result

def fourier_scale(rgb_whc_matrix, output_pixel_dim=(2048, 2048), projection_mode="visual"):
    # Per-channel 2-D spatial spectrum: transform only the spatial axes (0, 1), never the channel
    # axis, and fftshift only those axes so DC lands at the centre of both. Both spatial axes are
    # ordinary (full) Fourier axes, so the symmetric centre crop/pad in strip_and_pad is correct.
    spectrogram = torch.fft.fftshift(
        torch.fft.fft2(rgb_whc_matrix, dim=(0, 1)),
        dim=(0, 1),
    )

    #Either pad the spectrogram with 0s (upsample) or crop the outer high-frequency band (downsample)
    stripped_and_padded_spectrogram = strip_and_pad(spectrogram,
                                                    output_pixel_dim[0],
                                                    output_pixel_dim[1],
                                                    projection_mode)

    #Return the resampled spectrogram
    return stripped_and_padded_spectrogram

def fourier_reconstruct(spectrogram, output_pixel_dim, channels):
    # Invert fourier_scale: undo the spatial fftshift and inverse-transform the spatial axes only.
    # The channel axis is passed through untouched. Cropping/padding a spectrum breaks its exact
    # Hermitian symmetry, so take the real part (any residual imaginary component is numerical noise).
    expected_shape = (int(output_pixel_dim[0]), int(output_pixel_dim[1]), channels)
    if tuple(spectrogram.shape) != expected_shape:
        raise ValueError(f"Expected spectrogram of shape {expected_shape}, got {tuple(spectrogram.shape)}")
    return torch.fft.ifft2(
        torch.fft.ifftshift(spectrogram, dim=(0, 1)),
        dim=(0, 1),
    ).real

class ScaledRepresentation:
    def __init__(self, internal_gsd=(1.0, 1.0), internal_representation_dim=(2048, 2048)):
        self._internal_x_gsd = internal_gsd[0]
        self._internal_y_gsd = internal_gsd[1]
        self._internal_representation_x = internal_representation_dim[0]
        self._internal_representation_y = internal_representation_dim[1]
        self._x_axis_internal = torch.fft.fftshift(torch.fft.fftfreq(self._internal_representation_x, self._internal_x_gsd))
        self._y_axis_internal = torch.fft.fftshift(torch.fft.fftfreq(self._internal_representation_y, self._internal_y_gsd))
        self._internal_spacing = (self._x_axis_internal[1] - self._x_axis_internal[0], self._y_axis_internal[1] - self._y_axis_internal[0])

    def get_internal_freq_spacing(self):
        return self._internal_spacing

    def get_source_freq_spacing(self, rgb_whc_matrix, gsd):
        x_axis_source = torch.fft.fftshift(torch.fft.fftfreq(rgb_whc_matrix.shape[0], gsd[0]))
        y_axis_source = torch.fft.fftshift(torch.fft.fftfreq(rgb_whc_matrix.shape[1], gsd[1]))
        return (x_axis_source[1] - x_axis_source[0], y_axis_source[1] - y_axis_source[0])

    def get_representation(self, rgb_whc_matrix, projection_mode="visual"):
        return fourier_scale(rgb_whc_matrix,
                             output_pixel_dim=(self._internal_representation_x, self._internal_representation_y),
                             projection_mode=projection_mode)
