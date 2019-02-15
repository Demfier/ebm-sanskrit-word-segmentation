# Word Segmentation and Morphological Tagging in Sanskrit Using Energy Based Models

Code for our paper: [Free as in Free Word Order: An Energy Based Model for Word Segmentation and Morphological Tagging in Sanskrit](https://arxiv.org/pdf/1809.01446.pdf), accepted at EMNLP 2018, Brussels, Belgium.

Please find the pre-trained model and other data files distributed on zenodo [[link]](https://zenodo.org/record/1035413/#.XGZGj7pKhCV). Some other helper modules too have been provided in this repo inside the `helpers and outdated` folder.

## Team members:
[Amrith Krishna](https://github.com/krishnamrith12), [Bishal Santra](https://github.com/bsantraigi), Sasi Prasanth Bandaru, [Gaurav Sahu](https://github.com/demfier), [Vishnu Dutt Sharma](https://github.com/VishnuDuttSharma), Pavankumar Satuluri and [Pawan Goyal](https://github.com/pawangiitkgp).

## Getting Started

Please download the 2 compressed files 'dir.zip' and 'wordsegmentation.rar' to your working directory and extract them into folders named 'dir' and 'wordsegmentation' respectively.

Your working directory should be as follows
* Working Directory
  * wordsegmentation
    * skt_dcs_DS.bz2_4K_bigram_mir_10K
    * skt_dcs_DS.bz2_4K_bigram_mir_heldout
  * dir

## Prerequisites
* Python3
  * scipy
  * numpy
  * csv
  * pickle
  * multiprocessing
  * bz2
## Instructions for Training
Change your current directory to 'dir'

Run the file `Train_clique.py` by using the following command

* ```python Train_clique.py```

To train on different input features like BM2,BM3,BR2,BR3,PM2,PM3,PR,PR3 please modify the bz2_input_folder value in the main function before beginning the training.

Feature  | bz2_input_folder
------------- | -------------
BM2 | wordsegmentation/skt_dcs_DS.bz2_4K_bigram_mir_10K/
BM3 | wordsegmentation/skt_dcs_DS.bz2_1L_bigram_mir_10K
BR2 | wordsegmentation/skt_dcs_DS.bz2_4K_bigram_rfe_10K/
BR3 | wordsegmentation/skt_dcs_DS.bz2_1L_bigram_rfe_10K/
PM2 | wordsegmentation/skt_dcs_DS.bz2_4K_pmi_mir_10K/
PM3 | wordsegmentation/skt_dcs_DS.bz2_1L_pmi_mir_10K2/
PR2 | wordsegmentation/skt_dcs_DS.bz2_4K_pmi_rfe_10K/
PR3 | wordsegmentation/skt_dcs_DS.bz2_1L_pmi_rfe_10K/

## Instructions for Testing

After training, please modify the 'modelList' dictionary  in 'test_clique.py' with the name of the neural network that has been saved during training. While testing for a feature, please provide the name of the neural net which was trained for the same feature.

We only provide the trained model for the feature BM2 which was our best performing feature. If the name of the neural net is not changed, then the testing will be performed on the pre-trained model for BM2 provided in outputs/train_t7978754709018

To test with a particular feature vector use the tag of the feature while execution

* `python test_clique.py -t <tag>`

For example:
  * `python test_clique.py -t BM2`

After finishing the testing please run the following command to see the precision and recall values for both the word and word++ prediction tasks

* `python evaluate.py <tag>`

For example:
  * `python evaluate.py BM2`

## Reference:

If you find any part of our code useful, please cite:

```
@article{krishna2018free,
  title={Free as in Free Word Order: An Energy Based Model for Word Segmentation and Morphological Tagging in Sanskrit},
  author={Krishna, Amrith and Santra, Bishal and Bandaru, Sasi Prasanth and Sahu, Gaurav and Sharma, Vishnu Dutt and Satuluri, Pavankumar and Goyal, Pawan},
  journal={arXiv preprint arXiv:1809.01446},
  year={2018}
}
```
