Dependencies: python3

-------------

-numpy

-scipy

-multiprocessing

-pickle

-csv





Directory Strucure

--------------------
- 
-wordsegmentation
   
	- skt_dcs_DS.bz2_4K_bigram_mir_10K
   
	- skt_dcs_DS.bz2_4K_bigram_mir_heldout

-outputs

-inputs

-All python, text files provided




Instructions for Training

--------------------------


Run the file Train_clique.py by using the following command

python Train_clique.py



To train on different input features like BM2,BM3,BR2,BR3,PM2,PM3,PR,PR3 please modify the bz2_input_folder value in the main function before beginning the training.


  bz2_input_folder = 'wordsegmentation/skt_dcs_DS.bz2_4K_bigram_mir_10K/'   #bm2

bz2_input_folder = 'wordsegmentation/skt_dcs_DS.bz2_1L_bigram_mir_10K/' #bm3
    
bz2_input_folder = 'wordsegmentation/skt_dcs_DS.bz2_4K_bigram_rfe_10K/' #br2

bz2_input_folder = 'wordsegmentation/skt_dcs_DS.bz2_1L_bigram_rfe_10K/'  #br3
    
bz2_input_folder = 'wordsegmentation/skt_dcs_DS.bz2_4K_pmi_mir_10K/'   #pm2

bz2_input_folder = 'wordsegmentation/skt_dcs_DS.bz2_1L_pmi_mir_10K2/'   #pm3
    
bz2_input_folder = 'wordsegmentation/skt_dcs_DS.bz2_4K_pmi_rfe_10K/'     #pr2
   
bz2_input_folder = 'wordsegmentation/skt_dcs_DS.bz2_1L_pmi_rfe_10K/'   #pr3  

  

Please note the Neural Network's directory name which is displayed during training
(for example t7978754709018 means that the neural net is saved in the location outputs/train_t7978754709018 )




Instructions for Testing

------------------------


After training, please modify the 'modelList' dictionary  in 'test_clique.py' with the name of the neural network that has been saved during training. While testing for a feature, please provide the name of the neural net which was trained
for the same feature.



To test with a particular feature vector use the tag of the feature while execution


python test_clique.py -t <tag>



For example: python test_clique.py -t BM2



Instructions for Evaluation

---------------------------


After testing, please run the code as follows


python evaluate.py <tag>



For example: python evaluate.py BM2