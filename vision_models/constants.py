from enum import Enum
from torchvision import transforms
from types import SimpleNamespace
import json

class SimplerNamespace(SimpleNamespace):
    def keys(self):
        return list(self.__dict__.keys())

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)

VISION_DATASETS = SimplerNamespace(
    IMAGENET_1K = dict(
        dataset="imagenet-1k", 
        datatype="images", 
        data_description="Imagenet ILSVRC 2012 Release, contains ~1.3M images in 1000 categories.",
        datasize="1.3M",
        data_num_classes=1000,
        train_size=1281167,
        test_size=50000,
        data_bib=[
            json.dumps('''@inproceedings{deng2009imagenet,
                title={Imagenet: A large-scale hierarchical image database},
                author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
                booktitle={2009 IEEE conference on computer vision and pattern recognition},
                pages={248--255},
                year={2009},
                organization={Ieee}
            }'''),
            json.dumps('''@inproceedings{deng2009imagenet,
                title={Imagenet: A large-scale hierarchical image database},
                author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
                booktitle={2009 IEEE conference on computer vision and pattern recognition},
                pages={248--255},
                year={2009},
                organization={Ieee}
            }''')
        ]
    ),
    IMAGENET_21K_FALL2011 = dict(
        dataset="imagenet-21k-Fall2011",
        datatype="images",
        data_description="Imagenet Full Release (Fall2011), contains ~12M images in 21841 categories.",
        datasize="11.8M",
        train_size=11797632, 
        test_size=None, # need to download to figure it out
        data_num_classes=21841,
        source="https://academictorrents.com/details/564a77c1e1119da199ff32622a1609431b9f1c47",
        notes="exact images of imagenet-1k are a subset of imagenet-22k (11797632 train images)",
        preprocessing=None,
        data_bib=[
            json.dumps('''@inproceedings{deng2009imagenet,
                title={Imagenet: A large-scale hierarchical image database},
                author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
                booktitle={2009 IEEE conference on computer vision and pattern recognition},
                pages={248--255},
                year={2009},
                organization={Ieee}
            }'''),
            json.dumps('''@inproceedings{deng2009imagenet,
                title={Imagenet: A large-scale hierarchical image database},
                author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
                booktitle={2009 IEEE conference on computer vision and pattern recognition},
                pages={248--255},
                year={2009},
                organization={Ieee}
            }''')
        ]
    ),
    IMAGENET_21K_WINTER2021 = dict(
        dataset="imagenet-21k-Winter2021",
        datatype="images",
        datasize=None, # download to determine
        data_description="Imagenet Full Release (Winter2021), contains ~12M images in 21841 categories. Some categories removed (relative to Fall2011 version), and faces blurred for privacy.",
        train_size=None, # download to determine
        data_num_classes=21841,
        source="https://image-net.org/download-images.php",
        notes="exact images of imagenet-1k are a subset of imagenet-22k (11797632 train images)",
        download="https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/dataset_preprocessing/processing_instructions.md",
        preprocessing=None,
        data_bib=[
            json.dumps('''@inproceedings{deng2009imagenet,
                title={Imagenet: A large-scale hierarchical image database},
                author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
                booktitle={2009 IEEE conference on computer vision and pattern recognition},
                pages={248--255},
                year={2009},
                organization={Ieee}
            }'''),
            json.dumps('''@inproceedings{deng2009imagenet,
                title={Imagenet: A large-scale hierarchical image database},
                author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
                booktitle={2009 IEEE conference on computer vision and pattern recognition},
                pages={248--255},
                year={2009},
                organization={Ieee}
            }''')
        ]
    ),
    IMAGENET_21KP_FALL2011 = dict(
        dataset="imagenet-21k-P-Fall2011",
        datatype="images", 
        data_description="Imagenet Full Release (Fall2011), preprocessed (Alibaba-MIIL) to contain ~12M samples in 11221 classes.",
        datasize="12M",
        train_size=11797632, 
        test_size=561052,
        data_num_classes=11221,
        source="https://academictorrents.com/details/564a77c1e1119da199ff32622a1609431b9f1c47",
        notes="exact images of imagenet-1k are a subset of imagenet-22k (11797632 train images)",
        preprocessing="https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/dataset_preprocessing/processing_instructions.md",
        data_bib=[
            json.dumps('''@inproceedings{deng2009imagenet,
                title={Imagenet: A large-scale hierarchical image database},
                author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
                booktitle={2009 IEEE conference on computer vision and pattern recognition},
                pages={248--255},
                year={2009},
                organization={Ieee}
            }'''),
            json.dumps('''@inproceedings{deng2009imagenet,
                title={Imagenet: A large-scale hierarchical image database},
                author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
                booktitle={2009 IEEE conference on computer vision and pattern recognition},
                pages={248--255},
                year={2009},
                organization={Ieee}
            }'''),
            json.dumps('''@article{ridnik2021imagenet,
              title={Imagenet-21k pretraining for the masses},
              author={Ridnik, Tal and Ben-Baruch, Emanuel and Noy, Asaf and Zelnik-Manor, Lihi},
              journal={arXiv preprint arXiv:2104.10972},
              year={2021}
            }''')
        ]
    ),
    IMAGENET_21KP_WINTER2021 = dict(
        dataset="imagenet-21k-P-Winter2021",
        datatype="images", 
        datasize="11M",
        data_description="Imagenet Full Release (Winter2021), preprocessed (Alibaba-MIIL) to contain ~11M samples in 10450 classes.",        
        train_size=11060223, 
        test_size=522500,
        data_num_classes=10450,
        source="https://image-net.org/download-images.php",
        preprocessing="https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/dataset_preprocessing/processing_instructions.md",
        data_bib=[
            json.dumps('''@inproceedings{deng2009imagenet,
                title={Imagenet: A large-scale hierarchical image database},
                author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
                booktitle={2009 IEEE conference on computer vision and pattern recognition},
                pages={248--255},
                year={2009},
                organization={Ieee}
            }'''),
            json.dumps('''@inproceedings{deng2009imagenet,
                title={Imagenet: A large-scale hierarchical image database},
                author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
                booktitle={2009 IEEE conference on computer vision and pattern recognition},
                pages={248--255},
                year={2009},
                organization={Ieee}
            }'''),
            json.dumps('''@article{ridnik2021imagenet,
              title={Imagenet-21k pretraining for the masses},
              author={Ridnik, Tal and Ben-Baruch, Emanuel and Noy, Asaf and Zelnik-Manor, Lihi},
              journal={arXiv preprint arXiv:2104.10972},
              year={2021}
            }''')
        ]
    )    
)    

class AUGPOLICY(Enum):
    WUSAUG = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2,1.)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    TORCHVISION = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2,1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])