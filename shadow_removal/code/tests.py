import unittest
import torch
import config
import warnings


class TestComponents(unittest.TestCase):
    """Test Components"""

    def setUp(self):
        warnings.simplefilter("ignore", category=DeprecationWarning)
        self.rand_img_tensor_1 = torch.randn(1, 3, 480, 640, device=config.DEVICE)
        self.rand_img_tensor_2 = torch.randn(1, 3, 480, 640, device=config.DEVICE)
        self.rand_batch_img_tensor = torch.randn(8, 3, 480, 640, device=config.DEVICE)

    def test_dataloader_shape(self):
        from dataloaders.testLoaders import TestDataISTD

        temp_dl = torch.utils.data.DataLoader(
            TestDataISTD(),
            batch_size=1,
            num_workers=config.N_WORKERS,
        )

        x = next(iter(temp_dl))

        x_shape = x["shadow_image"].shape
        self.assertEqual(
            x_shape,
            self.rand_img_tensor_1.shape,
            f"Output {x_shape} != Expected {self.rand_img_tensor_1.shape}",
        )

    def test_critic(self):
        from components.Critic.Critic import Critic

        model = Critic().to(config.DEVICE)

        op_shape = model(self.rand_img_tensor_1, self.rand_img_tensor_2).mean().shape

        self.assertEqual(
            op_shape,
            torch.Size([]),
            f"Output {op_shape} != Expected {torch.Size([])}",
        )

    def test_actor(self):
        from components.Actor.Unet import ActorUNet

        mdl = ActorUNet(3, 3).to(config.DEVICE)
        op_shape = mdl(self.rand_batch_img_tensor).shape
        self.assertEqual(
            op_shape,
            self.rand_batch_img_tensor.shape,
            f"Output {op_shape} != Expected {self.rand_batch_img_tensor.shape}",
        )


if __name__ == "__main__":
    unittest.main()
