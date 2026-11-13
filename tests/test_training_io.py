"""CPU checks for image export, short-run checkpoints, and preview layout."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


class TrainingIOTests(unittest.TestCase):
    def test_preview_export_clips_to_unsigned_rgb(self):
        for filename, scale in [('s2_registration.py', 255), ('s3_appearance.py', 1),
                                ('scene/dataset_readers.py', 1)]:
            tree = ast.parse((ROOT / filename).read_text())
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == 'fromarray'
                     and isinstance(node.func.value, ast.Name) and node.func.value.id == 'Image']
            call = next(node for node in calls if 'np.clip' in ast.unparse(node))
            values = np.array([[[-20., 128., 300.]]])
            container = values.transpose(2, 0, 1) / scale if filename.startswith('s2') else values
            result = eval(compile(ast.Expression(call), '<image-export>', 'eval'),
                          dict(np=np, Image=Image, container=container, masked_img=values))
            self.assertEqual(result.mode, 'RGB')
            np.testing.assert_array_equal(np.asarray(result), [[[0, 128, 255]]])

    def test_texture_preview_fits_small_and_large_images(self):
        tree = ast.parse((ROOT / 's3_appearance.py').read_text())
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                    and any(isinstance(child, ast.Assign) and isinstance(child.targets[0], ast.Name)
                            and child.targets[0].id == 'image_height' for child in node.body))
        start = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'image_height')
        code = compile(ast.Module(body=loop.body[start:start + 5], type_ignores=[]), '<preview>', 'exec')
        for height in (510, 1020, 1280):
            scope = dict(torch=torch, F=F, _t=torch.ones(3, 1024, 1024),
                         viewpoint_cam=SimpleNamespace(original_image=torch.zeros(3, height, 750)))
            exec(code, scope)
            self.assertEqual(scope['texture'].shape, (3, height, 1024))
            self.assertGreater(float(scope['texture'].sum()), 0)

    def test_epoch_checkpoint_includes_canonical_directory(self):
        tree = ast.parse((ROOT / 's3_appearance.py').read_text())
        epoch_loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                          and isinstance(node.target, ast.Name) and node.target.id == 'epoch')
        calls = [node.value for node in epoch_loop.body if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                 and node.value.func.attr == 'save_ckpt']
        self.assertTrue(any(not call.keywords for call in calls))
        self.assertTrue(any(any(keyword.arg == 'name' for keyword in call.keywords) for call in calls))
        forced_epochs = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Attribute) and ast.unparse(target) == 'args.epochs'
                                 for target in node.targets)]
        self.assertEqual(forced_epochs, [])


if __name__ == '__main__':
    unittest.main()
