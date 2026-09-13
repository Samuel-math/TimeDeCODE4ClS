import unittest
import torch
from models import PatchVQVAE, MaskedClassifier


class ModelTests(unittest.TestCase):
    def test_three_stages(self):
        torch.manual_seed(42)
        vq = PatchVQVAE(8, 32, 16)
        x = torch.randn(3, 32, 2)
        patches = vq.patches(x)
        rec, loss, ids = vq(patches)
        self.assertEqual(rec.shape, patches.shape)
        (rec.square().mean() + loss).backward()
        self.assertIsNotNone(vq.encoder[0].weight.grad)
        self.assertIsNotNone(vq.codebook.weight.grad)
        model = MaskedClassifier(16, 32, 4, 2, 3, layers=1)
        model.masked_loss(ids.detach(), .35).backward()
        self.assertIsNotNone(model.token_head.weight.grad)
        self.assertEqual(model(ids).shape, (3, 3))
        self.assertTrue(torch.isfinite(model(ids)).all())

    def test_mask_evaluation_repeatable(self):
        model = MaskedClassifier(16, 32, 4, 2, 3, layers=1).eval()
        ids = torch.randint(16, (3, 2, 4))
        a = model.masked_loss(ids, .35, torch.Generator().manual_seed(42))
        b = model.masked_loss(ids, .35, torch.Generator().manual_seed(42))
        torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
