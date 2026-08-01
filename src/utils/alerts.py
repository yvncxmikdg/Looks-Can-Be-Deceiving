from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

class Alerter:
    def __init__(self, alert_api_key_path, alert_channel_id, message_prefix=""):
        with open(alert_api_key_path, "r") as f:
            self._client = WebClient(token=f.readlines()[-1].strip())
        self._alert_channel_id = alert_channel_id
        self._message_prefix = message_prefix

    def getAlertChannelId(self):
        return self._alert_channel_id

    def setAlertChannelId(self, new_channel_id):
        self._alert_channel_id = new_channel_id

    def sendAlert(self, message):
        try:
            full_message = self._message_prefix + message
            self._client.chat_postMessage(channel=self._alert_channel_id, text=full_message)
        except SlackApiError as e:
            print(f"Alert Error: {e.response['error']}")
