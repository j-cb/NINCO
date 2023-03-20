from torch.utils.data.dataset import Dataset
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision.datasets.folder import pil_loader, default_loader
import os, csv
import data.paths_config as paths_config
import pandas as pd

class ImageDictDataset(Dataset):
    def __init__(
        self,
        file_dict: dict,
        paths_prefix: str = '',
        name=None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
    ) -> None:
            self.paths_prefix = paths_prefix
            abs_file_dict = {os.path.join(paths_prefix, location):c for location, c in file_dict.items()}
            self.abs_file_dict = abs_file_dict
            self.samples = list(abs_file_dict.items())
            self.transform = transform
            self.target_transform = target_transform
            self.classes = list(dict.fromkeys(abs_file_dict.values()))
            self.loader = loader
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            if name is not None:
                self.__name__ = name
            else:
                self.__name__ = self.__class__.__name__
            self.targets = [self.class_to_idx[c] for c in abs_file_dict.values()]
    
    def get_individual_class(self, c):
        class_index = self.class_to_idx[c]
        return [(i, self.samples[i][0]) for i,v in enumerate(self.targets) if v == class_index]
    
    def __getitem__(self, index: int) -> Any:
        path, target_name = self.samples[index]
        target = self.class_to_idx[target_name]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

        
    _repr_indent = 4

    def __repr__(self) -> str:
        head = f'Dataset {self.__class__.__name__} {self.__name__}'
        body = [f"Number of datapoints: {self.__len__()}"]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)
    

class ImageCSVDataset(ImageDictDataset):
    def __init__(
        self,
        image_table_csv: str,
        paths_prefix: str = paths_config.dset_location_dict['NINCO'],
        name=None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
    ) -> None:
        with open(image_table_csv, 'r') as csv_file:
            reader = csv.reader(csv_file)
            num_csv_colums = len(next(reader, None)) #skip table header
            if num_csv_colums == 1:
                image_dict = {fl[0]: name for fl in reader}
            elif num_csv_colums == 2:
                image_dict = {fl[0]: fl[1] for fl in reader}
            else:
                raise ValueError('The csv should have 1 or 2 columns.')
            if name is not None:
                pass_name = name
            else:
                pass_name = image_table_csv.split('/')[-1][:-4]
            super().__init__(file_dict=image_dict,
                             paths_prefix=paths_prefix,
                             name=pass_name,
                             transform=transform,
                             target_transform=target_transform,
                             loader=loader,
                            )
           
        
class NINCO(ImageCSVDataset):
    def __init__(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
    ) -> None:
        super().__init__(image_table_csv=paths_config.dataset_csvs['NINCO_OOD_classes'],
                         paths_prefix=paths_config.dset_location_dict['NINCO'],
                         name='NINCO_OOD_classes',
                         transform=transform,
                         target_transform=target_transform,
                         loader=loader,
                        )

        names_df = pd.read_csv(paths_config.naming_csvs['NINCO_class_names'])
        self.class_printnames = {}
        self.class_abbreviations = {}
        for index, row in names_df.iterrows():
            if row['benchmark status'] == '1':
                self.class_printnames[row['filename']] = row['printname']
                if not pd.isnull(row['printname add-on in parentheses']):
                    self.class_printnames[row['filename']] += f" ({row['printname add-on in parentheses']})"
                self.class_abbreviations[row['filename']] = row['table abbrv.']
        
        
class NINCOOODUnitTests(ImageCSVDataset):
    def __init__(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
    ) -> None:
        super().__init__(image_table_csv=paths_config.dataset_csvs['NINCO_OOD_unit_tests'],
                         paths_prefix=paths_config.dset_location_dict['NINCO'],
                         name='NINCO_OOD_unit_tests',
                         transform=transform,
                         target_transform=target_transform,
                         loader=loader,
                        )
        
        names_df = pd.read_csv(paths_config.naming_csvs['NINCO_class_names'])
        self.class_printnames = {}
        self.class_abbreviations = {}
        for index, row in names_df.iterrows():
            if row['benchmark status'] == 'unit test':
                self.class_printnames[row['filename']] = row['printname']
                if not pd.isnull(row['printname add-on in parentheses']):
                    self.class_printnames[row['filename']] += f" ({row['printname add-on in parentheses']})"
                self.class_abbreviations[row['filename']] = row['table abbrv.']
        
class NINCOPopularDatasetsSubsamples(ImageCSVDataset):
    def __init__(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
    ) -> None:
        super().__init__(image_table_csv=paths_config.dataset_csvs['NINCO_popular_datasets_subsamples'],
                         paths_prefix=paths_config.dset_location_dict['NINCO'],
                         name='NINCO_popular_datasets_subsamples',
                         transform=transform,
                         target_transform=target_transform,
                         loader=loader,
                        )