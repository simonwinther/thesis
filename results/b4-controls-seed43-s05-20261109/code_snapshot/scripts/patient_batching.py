"""Build full batches with distinct BraTS patients while retaining scan visits."""

from pathlib import Path
import re


def patient_id(subject):
    path = Path(subject)
    case = path.parent.name if path.name.endswith('.nii.gz') else path.name
    match = re.fullmatch(r'(BraTS-GLI-\d{5})-\d{3}', case)
    if match is None:
        raise ValueError(f'Expected a BraTS GLI case or scan path, got {subject!r}')
    return match[1]


class DistinctPatientBatchSampler:
    """Repair a random case permutation; only the final incomplete batch is dropped."""

    def __init__(self, sampler, patients, batch_size=4):
        self.sampler = sampler
        self.patients = tuple(patients)
        self.batch_size = batch_size
        if batch_size < 1 or len(set(self.patients)) < batch_size:
            raise ValueError('Not enough distinct patients for a full batch')

    def __len__(self):
        return len(self.sampler) // self.batch_size

    def __iter__(self):
        order = list(self.sampler)
        if len(set(order)) != len(order) or any(i < 0 or i >= len(self.patients) for i in order):
            raise ValueError('Expected distinct valid case indices from the sampler')
        width = self.batch_size
        stop = len(order) // width * width
        for start in range(0, stop, width):
            end, seen = start + width, set()
            for position in range(start, end):
                duplicate = self.patients[order[position]]
                if duplicate in seen:
                    occupied = {self.patients[order[i]] for i in range(start, end) if i != position}
                    replacement = next((i for i in range(end, len(order))
                                        if self.patients[order[i]] not in occupied), None)
                    if replacement is None:
                        # Repair a short final tail without dropping any extra visits.
                        for i in range(start):
                            earlier_start = i // width * width
                            earlier = {self.patients[order[j]]
                                       for j in range(earlier_start, earlier_start + width) if j != i}
                            if self.patients[order[i]] not in occupied and duplicate not in earlier:
                                replacement = i
                                break
                    if replacement is None:
                        raise ValueError('Cannot form full batches with distinct patients')
                    order[position], order[replacement] = order[replacement], order[position]
                seen.add(self.patients[order[position]])
        # Repair is completed before yielding, since a final-tail swap may change an earlier batch.
        for start in range(0, stop, width):
            batch = order[start:start + width]
            if len({self.patients[i] for i in batch}) != width:
                raise RuntimeError('Patient batch repair failed')
            yield batch
