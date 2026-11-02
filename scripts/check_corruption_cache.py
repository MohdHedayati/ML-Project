from pathlib import Path
import os
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval as evaluation
from dataset_loader import TrafficSignDataset
from models.baseline.model import Baseline
from noise import CachedCorruptedDataset, CorruptedDataset, corruption_cache_key


def main():
    torch.set_num_threads(1)
    with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary_directory:
        root = Path(temporary_directory)
        Image.new('RGB', (49, 49), (80, 100, 120)).save(root / 'sign.png')
        (root / 'Test.csv').write_text('Path,ClassId\nsign.png,0\nsign.png,1\n')
        dataset = TrafficSignDataset(split='test', root_dir=str(root))
        key = corruption_cache_key(dataset, (49, 49))
        assert key == corruption_cache_key(dataset, (49, 49))
        assert key != corruption_cache_key(dataset, (32, 32))
        dataset.labels[0] = 2
        assert key != corruption_cache_key(dataset, (49, 49))
        dataset.labels[0] = 0
        source = root / 'sign.png'
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        assert key != corruption_cache_key(dataset, (49, 49))
        corrupted = CorruptedDataset(dataset, noise_type='brightness')
        expected, label = corrupted[0]
        from noise import add_noise_and_scale
        with patch('noise.add_noise_and_scale', wraps=add_noise_and_scale) as generate:
            cached = CachedCorruptedDataset(corrupted, root / 'cache')
            assert generate.call_count == 2
        actual, actual_label = cached[0]
        assert torch.equal(expected, actual) and actual_label == label
        assert np.load(cached.path).dtype == np.uint8
        with patch('noise.add_noise_and_scale', side_effect=AssertionError('Cache miss')):
            reused = CachedCorruptedDataset(corrupted, root / 'cache')
            assert torch.equal(reused[0][0], actual)
            for workers in (0, 2):
                batches = list(DataLoader(reused, batch_size=2, num_workers=workers))
                assert batches[0][1].tolist() == [0, 1]
                assert torch.equal(batches[0][0][0], actual)
        cached.images = None
        reused.images = None
        cached.path.write_bytes(b'incomplete file')
        repaired = CachedCorruptedDataset(corrupted, root / 'cache')
        assert torch.equal(repaired[0][0], expected)
        repaired.images = None
        with patch('noise.add_noise_and_scale', side_effect=KeyboardInterrupt):
            try:
                CachedCorruptedDataset(corrupted, root / 'interrupted')
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError('Expected interruption')
        assert not list((root / 'interrupted').iterdir())
        retry = CachedCorruptedDataset(corrupted, root / 'interrupted', num_workers=2)
        assert torch.equal(retry[0][0], expected)
        retry.images = None

        engine = evaluation.Evaluator(device='cpu', num_workers=0)
        model = Baseline(43)
        adaptation_images = []

        def adapt(model, loader):
            adaptation_images.append(next(iter(loader))[0])

        with patch.object(evaluation, '__file__', str(root / 'eval.py')), \
                patch.object(evaluation, 'NOISE_TYPES', ['brightness']), \
                patch.object(engine, '_save_severity_confusion_matrix'), \
                patch.object(engine, 'plot_robustness'):
            first = engine.test_with_noise(model, dataset, str(root / 'first'), batch_size=2, pre_test_hook=adapt)
            with patch('noise.add_noise_and_scale', side_effect=AssertionError('Regenerated cached images')):
                second = engine.test_with_noise(model, dataset, str(root / 'second'), batch_size=2, pre_test_hook=adapt)
            assert first.equals(second)
            assert len(adaptation_images) == 10
            assert all(torch.equal(adaptation_images[index], adaptation_images[index + 5]) for index in range(5))
            assert len(list((root / 'output' / '.corruption_cache').rglob('*.npy'))) == 5
    print('PASS: cache reuse, exact pixels, invalidation, workers, damaged files, interruption, and evaluator integration.')


if __name__ == '__main__':
    main()
