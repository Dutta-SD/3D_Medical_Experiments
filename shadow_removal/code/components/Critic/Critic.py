import torch
import config


class Critic(torch.nn.Module):
    def __init__(self, single_image_in_channel=3, **kwargs):
        super(Critic, self).__init__()

        # TODO -- Hardcoded 140, Modify
        self._layer = torch.nn.Sequential(
            torch.nn.Conv2d(
                2 * single_image_in_channel, 16, kernel_size=5, padding="same"
            ),
            torch.nn.MaxPool2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 8, 3, padding="same"),
            torch.nn.MaxPool2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 2, 3, padding="same"),
            torch.nn.MaxPool2d(4),
            torch.nn.Flatten(),
            torch.nn.Linear(140, 1, bias=False),
        )

    def forward(self, image_with_generated_shadow, image_shadow_free):
        final_tensor = torch.cat(
            (image_shadow_free, image_with_generated_shadow), dim=1
        )
        return self._layer(final_tensor)
