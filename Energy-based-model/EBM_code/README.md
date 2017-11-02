<div id="readme" class="readme blob instapaper_body">
    <article class="markdown-body entry-content" itemprop="text"><h1><a href="#word-segmentation-in-sanskrit-using-energy-based-models" aria-hidden="true" class="anchor" id="user-content-word-segmentation-in-sanskrit-using-energy-based-models"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Word Segmentation in Sanskrit Using Energy Based Models</h1>
<p>This is the repo containing the codes, and instructions for training, testing and evaluating word segmentation code in sanskrit using energy based model (EBM). However, the data for the task isn't released yet.</p>
<h2><a href="#getting-started" aria-hidden="true" class="anchor" id="user-content-getting-started"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Getting Started</h2>
<p>Please download all the contents of this repository to your working directory
Extract the rar files as folders into the working directory</p>
<p>Your working directory should be as follows</p>
<ul>
<li>Working Directory
<ul>
<li>wordsegmentation
<ul>
<li>skt_dcs_DS.bz2_4K_bigram_mir_10K</li>
<li>skt_dcs_DS.bz2_4K_bigram_mir_heldout</li>
</ul>
</li>
<li>outputs
<ul>
<li>train_t7978754709018</li>
</ul>
</li>
<li>Train_clique.py</li>
<li>test_clique.py</li>
<li>all other python files and text files provided in the repo</li>
</ul>
</li>
</ul>
<h2><a href="#prerequisites" aria-hidden="true" class="anchor" id="user-content-prerequisites"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Prerequisites</h2>
<ul>
<li>Python3
<ul>
<li>scipy</li>
<li>numpy</li>
<li>csv</li>
<li>pickle</li>
<li>multiprocessing</li>
<li>bz2</li>
</ul>
</li>
</ul>
<h2><a href="#instructions-for-training" aria-hidden="true" class="anchor" id="user-content-instructions-for-training"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Instructions for Training</h2>
<p>Run the file Train_clique.py by using the following command</p>
<ul>
<li>python Train_clique.py</li>
</ul>
<p>To train on different input features like BM2,BM3,BR2,BR3,PM2,PM3,PR,PR3 please modify the bz2_input_folder value in the main function before beginning the training.</p>
<table>
<thead>
<tr>
<th>Feature</th>
<th>bz2_input_folder</th>
</tr>
</thead>
<tbody>
<tr>
<td>BM2</td>
<td>wordsegmentation/skt_dcs_DS.bz2_4K_bigram_mir_10K/</td>
</tr>
<tr>
<td>BM3</td>
<td>wordsegmentation/skt_dcs_DS.bz2_1L_bigram_mir_10K</td>
</tr>
<tr>
<td>BR2</td>
<td>wordsegmentation/skt_dcs_DS.bz2_4K_bigram_rfe_10K/</td>
</tr>
<tr>
<td>BR3</td>
<td>wordsegmentation/skt_dcs_DS.bz2_1L_bigram_rfe_10K/</td>
</tr>
<tr>
<td>PM2</td>
<td>wordsegmentation/skt_dcs_DS.bz2_4K_pmi_mir_10K/</td>
</tr>
<tr>
<td>PM3</td>
<td>wordsegmentation/skt_dcs_DS.bz2_1L_pmi_mir_10K2/</td>
</tr>
<tr>
<td>PR2</td>
<td>wordsegmentation/skt_dcs_DS.bz2_4K_pmi_rfe_10K/</td>
</tr>
<tr>
<td>PR3</td>
<td>wordsegmentation/skt_dcs_DS.bz2_1L_pmi_rfe_10K/</td>
</tr></tbody></table>
<h2><a href="#instructions-for-testing" aria-hidden="true" class="anchor" id="user-content-instructions-for-testing"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Instructions for Testing</h2>
<p>After training, please modify the 'modelList' dictionary  in 'test_clique.py' with the name of the neural network that has been saved during training. While testing for a feature, please provide the name of the neural net which was trained
for the same feature.</p>
<p>To test with a particular feature vector use the tag of the feature while execution</p>
<ul>
<li>python test_clique.py -t </li>
</ul>
<p>For example:</p>
<ul>
<li>python test_clique.py -t BM2</li>
</ul>
<p>After finishing the testing please run the following command to see the precision and recall values for both the word and word++ prediction tasks</p>
<ul>
<li>python evaluate.py </li>
</ul>
<p>For example:</p>
<ul>
<li>python evaluate.py BM2</li>
</ul>
