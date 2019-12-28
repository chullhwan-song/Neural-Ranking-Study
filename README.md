
# Introduction

##### 참조
* [링크](https://github.com/NTMC-Community/awesome-neural-models-for-semantic-match/blob/master/README.md) : modified
* [링크](https://github.com/brylevkirill/notes/blob/master/Information%20Retrieval.md)

## Task
<table>
<tr>
<th width=30%, bgcolor=#999999 >Tasks</th> 
<th width=20%, bgcolor=#999999>Source </th>
<th width="20%", bgcolor=#999999>Target </th>
<th width="20%", bgcolor=#999999>비고 </th>
</tr>
<tr>
<td align="center", bgcolor=#eeeeee> Ad-hoc Information Retrieval : Text </td>
<td align="center", bgcolor=#eeeeee> Text query </td>
<td align="center", bgcolor=#eeeeee> Text document (title/content) </td>
<td align="center", bgcolor=#eeeeee> 발빠르게 SOTA논문을 구현하고 적용고민 </td> 
</tr>
<tr>
<td align="center", bgcolor=#eeeeee> Ad-hoc Information Retrieval : Multi-Modal </td>
<td align="center", bgcolor=#eeeeee> Text query </td>
<td align="center", bgcolor=#eeeeee> Image or Video or Text document (title/content) </td>
<td align="center", bgcolor=#eeeeee> 기존 Ad-hoc Information Retrieval 연구등에서 리뷰하고 관련 아디이어를 내어 고민해본다.  </td> 
</tr>
</table>

* Ad-hoc Information Retrieval : Multi-Modal 
  * 거의 없는관계로 관련 유사 Multi-Modal embedding(VQA, Image to Text Matching..)/L2R/GNN등의 연구등을 살펴보고 Idea를 내본다.  

## Ad-hoc Information Retrieval ??

---

**Information retrieval** (**IR**) is the activity of obtaining information system resources relevant to an information need from a collection. Searches can be based on full-text or other content-based indexing.  Here, the **Ad-hoc information retrieval** refer in particular to text-based retrieval where documents in the collection remain relative static and new queries are submitted to the system continually (cited from the [survey](https://arxiv.org/pdf/1903.06902.pdf)).


# Ad-hoc Information Retrieval using Neural Ranking
* Learning to Rank

## Data

the number of queries is huge. Some benchmark datasets are listed in the following,

* [**Robust04**](https://trec.nist.gov/data/t13_robust.html) is a small news dataset which contains about 0.5 million documents in total. The queries are collected from TREC Robust Track 2004. There are 250 queries in total.

* [**Cluebweb09**](https://trec.nist.gov/data/webmain.html) is a large Web collection which contains about 34 million documents in total. The queries are accumulated from TREC Web Tracks 2009, 2010, and 2011. There are 150 queries in total.

* [**Gov2**](https://trec.nist.gov/data/terabyte.html) is a large Web collection where the pages are crawled from .gov. It consists of 25 million documents in total. The queries are accumulated over TREC Terabyte Tracks 2004, 2005, and 2006. There are 150 queries in total.

* [**MSMARCO Passage Reranking**](http://www.msmarco.org/dataset.aspx) provides a large number of information question-style queries from Bing's search logs. There passages are annotated by humans with relevant/non-relevant labels. There are 8,841822 documents in total. There are 808,731queries, 6,980 queries and 48,598 queries for train, validation and test, respectively.
 
* [**LETOR**](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/) is a package of benchmark data sets for research on LEarning TO Rank, which contains standard features, relevance judgments, data partitioning, evaluation tools, and several baselines. Version 1.0 was released in April 2007. Version 2.0 was released in Dec. 2007. Version 3.0 was released in Dec. 2008. This version, 4.0, was released in July 2009. Very different from previous versions (V3.0 is an update based on V2.0 and V2.0 is an update based on V1.0), LETOR4.0 is a totally new release. It uses the Gov2 web page collection (~25M pages) and two query sets from Million Query track of TREC 2007 and TREC 2008. We call the two query sets **MQ2007** and **MQ2008** for short. There are about 1700 queries in MQ2007 with labeled documents and about 800 queries in MQ2008 with labeled documents.

* [**OHSUMED**](http://mlr.cs.umass.edu/ml/machine-learning-databases/ohsumed/)([or link](http://davis.wpi.edu/xmdv/datasets/ohsumed.html)) test collection is a set of 348,566 references from MEDLINE, the on-line medical information database, consisting of titles and/or abstracts from 270 medical journals over a five-year period (1987-1991). The available fields are title, abstract, MeSH indexing terms, author, source, and publication type. The National Library of Medicine has agreed to make the MEDLINE references in the test database available for experimentation, restricted to the following conditions:
1. The data will not be used in any non-experimental clinical, library, or other setting.
2. Any human users of the data will explicitly be told that the data is incomplete and out-of-date.
(From the following source)

## Paper : Text based Embedding

[1] **Word2Vec**

* paper:  [[paper]](https://arxiv.org/abs/1301.3781)
* review: [[review]](https://github.com/chullhwan-song/Reading-Paper/issues/36)

Efficient Estimation of Word Representations in Vector Space  : 

[2] **node2vec** 

* paper: [[paper]](https://cs.stanford.edu/people/jure/pubs/node2vec-kdd16.pdf)
* review: [[review]](https://github.com/chullhwan-song/Reading-Paper/issues/50)

node2vec: Scalable Feature Learning for Networks  

[3] **BERT**

* paper: [[paper]](https://arxiv.org/abs/1810.04805)
* review: [[review]](https://github.com/chullhwan-song/Reading-Paper/issues/202)(~ing)

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding 

[4]  **DESM**

* source code :[[code]](https://www.kaggle.com/girianantharaman/dual-embeddings-space-model-demo) [[[code]](https://github.com/bmitra-msft/Demos/blob/master/notebooks/DESM.ipynb) & [[pre_trained_model_from_MS]](https://www.microsoft.com/en-us/download/details.aspx?id=52597)]
* paper :[[pdf]](https://arxiv.org/abs/1602.01137) [[pdf]](https://dl.acm.org/citation.cfm?id=2889361)
* review :[[review]](https://www.slideshare.net/BhaskarMitra3/dual-embedding-space-model-desm) [[review]](http://cips-upload.bj.bcebos.com/2017/ssatt2017/ATT2017-IRI.pdf) [[review]](https://web.stanford.edu/class/cs276/handouts/lecture20-distributed-representations.pdf) [[review]](http://nn4ir.com/sigir2017/slides/04_TextMatchingII.pdf)

A Dual Embedding Space Model for Document Ranking. *2016*

Improving Document Ranking with Dual Word Embeddings



## Paper : MulitModel (image to image(Video)) based Embedding

* DeViSE: A Deep Visual-Semantic Embedding Model : [[paper]](https://research.google.com/pubs/pub41869.html)[[review]](https://github.com/chullhwan-song/Reading-Paper/issues/1)
* Dual Attention Networks for Multimodal Reasoning and Matching : [[paper]](https://www.google.co.kr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=0ahUKEwiOl5Pj19LUAhVKvLwKHVpoDdcQFggvMAE&url=https%3A%2F%2Farxiv.org%2Fpdf%2F1611.00471&usg=AFQjCNEkNnTcTYyq7AI9uFuQKDHom0ai1w)[[review]](https://github.com/chullhwan-song/Reading-Paper/issues/21)
* Learning Deep Structure-Preserving Image-Text Embeddings : [[paper]](https://arxiv.org/abs/1511.06078)[[review]](https://github.com/chullhwan-song/Reading-Paper/issues/26)
* Learning Two-Branch Neural Networks for Image-Text Matching Tasks : [[paper]](https://arxiv.org/abs/1810.02443)[[link_review]](https://github.com/chullhwan-song/Reading-Paper/issues/56)

## Paper : Neural information retrieval (NeuIR)

### Evaluation

* 이쪽분야가 benchmarkset이 정립이안되는듯한 모습이 보인다.(제생각)
* 일단 LETOR로.그나마..

##### MQ2007
<table>
<tr>
<th width=30%, bgcolor=#999991 >Paper</th> 
<th width=30%, bgcolor=#999999 >Year </th>
<th width=20%, bgcolor=#999999>P@1</th>
<th width="20%", bgcolor=#999999>P@5</th>
<th width=20%, bgcolor=#999999>P@10</th>
<th width="20%", bgcolor=#999999>NDCG@1</th>
<th width="20%", bgcolor=#999999>NDCG@5</th>
<th width="20%", bgcolor=#999999>NDCG@10</th>
<th width="20%", bgcolor=#999999>MAP</th>
</tr>
<tr>
<td align="left", bgcolor=#eeeeee> BM25 </td>
<td align="center", bgcolor=#eeeeee> 1994 </td>
<td align="center", bgcolor=#eeeeee> 0.427 </td>
<td align="center", bgcolor=#eeeeee> 0.388 </td>
<td align="center", bgcolor=#eeeeee> 0.358 </td>
<td align="center", bgcolor=#eeeeee> 0.366 </td>
<td align="center", bgcolor=#eeeeee> 0.384 </td>
<td align="center", bgcolor=#eeeeee> 0.414 </td>
<td align="center", bgcolor=#eeeeee> 0.450 </td>
</tr>
<tr>
<td align="left", bgcolor=#eeeeee> [1]DSSM </td>
<td align="center", bgcolor=#eeeeee> 2013</td>
<td align="center", bgcolor=#eeeeee> 0.345 </td>
<td align="center", bgcolor=#eeeeee> 0.359 </td>
<td align="center", bgcolor=#eeeeee> 0.352 </td>
<td align="center", bgcolor=#eeeeee> 0.290 </td>
<td align="center", bgcolor=#eeeeee> 0.335 </td>
<td align="center", bgcolor=#eeeeee> 0.371 </td>
<td align="center", bgcolor=#eeeeee> 0.409 </td>
</tr>
<tr>
<td align="left", bgcolor=#eeeeee> [3]DRMM </td>
<td align="center", bgcolor=#eeeeee> 2016 </td>
<td align="center", bgcolor=#eeeeee> 0.450 </td>
<td align="center", bgcolor=#eeeeee> 0.417 </td>
<td align="center", bgcolor=#eeeeee> 0.388 </td>
<td align="center", bgcolor=#eeeeee> 0.380 </td>
<td align="center", bgcolor=#eeeeee> 0.408 </td>
<td align="center", bgcolor=#eeeeee> 0.440 </td>
<td align="center", bgcolor=#eeeeee> 0.467 </td>
</tr>
<tr>
<td align="left", bgcolor=#eeeeee> [6]Duet </td>
<td align="center", bgcolor=#eeeeee> 2017 </td>
<td align="center", bgcolor=#eeeeee> 0.473 </td>
<td align="center", bgcolor=#eeeeee> 0.428 </td>
<td align="center", bgcolor=#eeeeee> 0.398 </td>
<td align="center", bgcolor=#eeeeee> 0.409 </td>
<td align="center", bgcolor=#eeeeee> 0.431 </td>
<td align="center", bgcolor=#eeeeee> 0.453 </td>
<td align="center", bgcolor=#eeeeee> 0.474 </td>
</tr>
<td align="left", bgcolor=#eeeeee> [10]DeepRank </td>
<td align="center", bgcolor=#eeeeee> 2017 </td>
<td align="center", bgcolor=#eeeeee> 0.508 </td>
<td align="center", bgcolor=#eeeeee> 0.452 </td>
<td align="center", bgcolor=#eeeeee> 0.412 </td>
<td align="center", bgcolor=#eeeeee> 0.441 </td>
<td align="center", bgcolor=#eeeeee> 0.457 </td>
<td align="center", bgcolor=#eeeeee> 0.482 </td>
<td align="center", bgcolor=#eeeeee> 0.497 </td>
</tr>
<td align="left", bgcolor=#eeeeee> [11]HiNT </td>
<td align="center", bgcolor=#eeeeee> 2018 </td>
<td align="center", bgcolor=#eeeeee> <b> 0.515 </td>
<td align="center", bgcolor=#eeeeee> <b> 0.461 </td>
<td align="center", bgcolor=#eeeeee> <b> 0.418 </td>
<td align="center", bgcolor=#eeeeee> <b>0.447 </td>
<td align="center", bgcolor=#eeeeee> <b>0.463 </td>
<td align="center", bgcolor=#eeeeee> <b>0.490 </td>
<td align="center", bgcolor=#eeeeee> <b>0.502 </td>
</tr>
</table>

#####  MQ2008

<table>
<tr>
<th width=30%, bgcolor=#999999 >Paper</th> 
<th width=30%, bgcolor=#999999 >Year </th>
<th width=20%, bgcolor=#999999>P@1</th>
<th width="20%", bgcolor=#999999>P@5</th>
<th width=20%, bgcolor=#999999>P@10</th>
<th width="20%", bgcolor=#999999>NDCG@1</th>
<th width="20%", bgcolor=#999999>NDCG@5</th>
<th width="20%", bgcolor=#999999>NDCG@10</th>
<th width="20%", bgcolor=#999999>MAP</th>
</tr>
<tr> 
<td align="left", bgcolor=#eeeeee> BM25 </td>
<td align="center", bgcolor=#eeeeee> 1994 </td>
<td align="center", bgcolor=#eeeeee> 0.408 </td>
<td align="center", bgcolor=#eeeeee> 0.337 </td>
<td align="center", bgcolor=#eeeeee> 0.245 </td>
<td align="center", bgcolor=#eeeeee> 0.344 </td>
<td align="center", bgcolor=#eeeeee> 0.461 </td>
<td align="center", bgcolor=#eeeeee> 0.220 </td>
<td align="center", bgcolor=#eeeeee> 0.465 </td>
</tr>
<tr> 
<td align="left", bgcolor=#eeeeee> [1]DSSM </td>
<td align="center", bgcolor=#eeeeee> 2013</td>
<td align="center", bgcolor=#eeeeee> 0.341 </td>
<td align="center", bgcolor=#eeeeee> 0.284 </td>
<td align="center", bgcolor=#eeeeee> 0.221 </td>
<td align="center", bgcolor=#eeeeee> 0.286 </td>
<td align="center", bgcolor=#eeeeee> 0.378 </td>
<td align="center", bgcolor=#eeeeee> 0.178 </td>
<td align="center", bgcolor=#eeeeee> 0.391 </td>
</tr>
<tr>
<td align="left", bgcolor=#eeeeee> [3]DRMM </td>
<td align="center", bgcolor=#eeeeee> 2016 </td>
<td align="center", bgcolor=#eeeeee> 0.450 </td>
<td align="center", bgcolor=#eeeeee> 0.337 </td>
<td align="center", bgcolor=#eeeeee> 0.242 </td>
<td align="center", bgcolor=#eeeeee> 0.381 </td>
<td align="center", bgcolor=#eeeeee> 0.466 </td>
<td align="center", bgcolor=#eeeeee> 0.219 </td>
<td align="center", bgcolor=#eeeeee> 0.473 </td>
</tr>
<tr>
<td align="left", bgcolor=#eeeeee> [6]Duet </td>
<td align="center", bgcolor=#eeeeee> 2017 </td>
<td align="center", bgcolor=#eeeeee> 0.452 </td>
<td align="center", bgcolor=#eeeeee> 0.341 </td>
<td align="center", bgcolor=#eeeeee> 0.240 </td>
<td align="center", bgcolor=#eeeeee> 0.385 </td>
<td align="center", bgcolor=#eeeeee> 0.471 </td>
<td align="center", bgcolor=#eeeeee> 0.216 </td>
<td align="center", bgcolor=#eeeeee> 0.476 </td>
</tr> 
<td align="left", bgcolor=#eeeeee> [10]DeepRank </td>
<td align="center", bgcolor=#eeeeee> 2017 </td>
<td align="center", bgcolor=#eeeeee> 0.482 </td>
<td align="center", bgcolor=#eeeeee> 0.359 </td>
<td align="center", bgcolor=#eeeeee> 0.252 </td>
<td align="center", bgcolor=#eeeeee> 0.406 </td>
<td align="center", bgcolor=#eeeeee> 0.496 </td>
<td align="center", bgcolor=#eeeeee> 0.240 </td>
<td align="center", bgcolor=#eeeeee> 0.498 </td>
</tr> 
<td align="left", bgcolor=#eeeeee> [11]HiNT </td>
<td align="center", bgcolor=#eeeeee> 2018 </td>
<td align="center", bgcolor=#eeeeee> <b>0.491 </td>
<td align="center", bgcolor=#eeeeee> <b>0.367 </td>
<td align="center", bgcolor=#eeeeee> <b>0.255 </td>
<td align="center", bgcolor=#eeeeee> <b>0.415 </td>
<td align="center", bgcolor=#eeeeee> <b>0.501 </td>
<td align="center", bgcolor=#eeeeee> <b> 0.244 </td>
<td align="center", bgcolor=#eeeeee> <b>0.505 </td>
</tr>
</table>

#####  Ad-Hoc Information Retrieval on TREC Robust04 
[링크](https://paperswithcode.com/sota/ad-hoc-information-retrieval-on-trec-robust04)

### Papers

#### [1] **DSSM** 

* source code : [[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/dssm.py) [[code]](https://github.com/zheng5yu9/siamese_dssm) [[code]](https://github.com/mingspy/cnn-dssm) [[code]](https://github.com/xubaochuan/dssm)[[code]](https://github.com/sunnyuanovo/DSSM) [[code]](https://github.com/songyandong/dssm-lstm)   [[code]](https://github.com/baharefatemi/DSSM) [[code]](https://github.com/ShuaiyiLiu/sent_cnn_tf)  [[code]](https://github.com/pengming617/text_matching/blob/master/dssm/dssm_model.py) [[code]](https://github.com/ChenglongChen/tensorflow-DSMM)  [[code]](https://github.com/kn45/ltr-dnn) [[code]](https://github.com/shelldream/DSSM)
* paper : [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf) 
* review : [[tutorial]](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/07/dl-summer-school-2017.-Jianfeng-Gao.v2.pdf) 

**Learning Deep Structured Semantic Models for Web Search using Clickthrough Data.** *CIKM 2013*.


[2] **CDSSM**  

* source code :[[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/cdssm.py) [[code]](https://github.com/Sherriiie/Cdssm) [[code]](https://github.com/liuqiangict/CDSSM_QK)  [[code]](https://github.com/zhijieqiu/tfLearnCDSSM)
* paper :[[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/www2014_cdssm_p07.pdf) 


**Learning Semantic Representations Using Convolutional Neural Networks for Web Search.** *WWW 2014*.
**A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval.** *CIKM 2014*.


[3] **DRMM** 

* source code :[[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/drmm.py)  [[code]](https://github.com/minhtannguyen/drmm-tensorflow) [[code]](https://github.com/EmanueleC/DRMM_repro)  [[code]](https://github.com/DengXuedx/tensorflow-DRMM) [[code]](https://github.com/DengXuedx/tensorflow-DRMM)
* paper :[[pdf]](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) 

**A Deep Relevance Matching Model for Ad-hoc Retrieval.** *DRMM 2016*.

[4] **KNRM**  

* source code :[[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/knrm.py) [[code]](https://github.com/jyy0553/KNRM)[[code]](https://github.com/ChengjinLi/knrm)
* paper :[[pdf]](https://arxiv.org/pdf/1706.06613.pdf) 
* review : [[review]](https://github.com/chullhwan-song/Reading-Paper/issues/258)

**End-to-End Neural Ad-hoc Ranking with Kernel Pooling.** *SIGIR 2017*

* source code :[[code]](https://github.com/peternara/K-NRM) 


[5] **CONV-KNRM**  

* source code :[[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/conv_knrm.py) [[code]](https://github.com/yunhenk/Conv-KNRM) [[code]](https://github.com/jyy0553/Conv-KNRM) [[code]](https://github.com/shuyanzhou/aggregated_semantic_matching) [[code]](https://github.com/Alchemist75/KNRM_FZ) [[code]](https://github.com/peternara/Conv-KNRM-ranking)
* paper :[[pdf]](http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf)
* review : [[review]](https://github.com/chullhwan-song/Reading-Paper/issues/257)

**Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search.** *WSDM 2018*

[6] **Duet** 

* source code : [[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/duet.py) 
* paper :[[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf) 

**Learning to Match using Local and Distributed Representations of Text for Web Search.** *WWW 2017*

[7] **Co-PACRR**  

* source code :[[code]](https://github.com/bamdart/Co-PACRR) 
* paper :[[pdf]](https://arxiv.org/pdf/1706.10192.pdf) 

**Co-PACRR: A Context-Aware Neural IR Model for Ad-hoc Retrieval.** *WSDM 2018*.

[8] **LSTM-RNN** 

* source code :[[code not ready]]()
* paper :[[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/02/LSTM_DSSM_IEEE_TASLP.pdf) 


**Deep Sentence Embsedding Using Long Short-Term Memory Networks: Analysis and Application to Information Retrieval.** *TASLP 2016*.

[9] **DRMM_TKS** 

* source code : [[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/drmm_tks.py) 
* paper :[[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-01012-6_2) 



**A Deep Relevance Matching Model for Ad-hoc Retrieval (*A variation of DRMM).** *CCIR 2018*.

[10] **DeepRank**  

* source code :[[code]]() 
* paper :[[pdf]](https://arxiv.org/pdf/1710.05649.pdf) 
* review: [[review]](https://github.com/chullhwan-song/Reading-Paper/issues/52)

**DeepRank: A New Deep Architecture for Relevance Ranking in Information Retrieval.** *CIKM 2017*

[11] **HiNT**  

* source code : [[code]](https://github.com/faneshion/HiNT) 
* paper :[[pdf]](https://arxiv.org/pdf/1805.05737.pdf) 


**Modeling Diverse Relevance Patterns in Ad-hoc Retrieval.** *SIGIR 2018*.


### adding

[12] **snrm**  

* source code : [[code]](https://github.com/hamed-zamani/snrm) 
* paper : [[pdf]](https://ciir-publications.cs.umass.edu/getpdf.php?id=1302) 
* review : [[review]](https://github.com/chullhwan-song/Reading-Paper/issues/162) [[review]](https://mostafadehghani.com/wp-content/uploads/2018/11/benelearn_2018.pdf) [[review]](https://oss.navercorp.com/vl/NextImageSearch_Study/files/144603/From.Neural.Re-Ranking.to.Neural.Ranking.pdf)

**From Neural Re-Ranking to Neural Ranking: Learning a Sparse Representation for Inverted Indexing.** *ACM 2018*


[13] **RankNet** 

* source code :[[code]](https://github.com/mzhang001/tfranknet) [[code_cpp]](https://github.com/kevinking/Ranklibc) [[code]](https://github.com/rkamio/ranknet-tensorflow) 
* paper :[[pdf]](https://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf) 

**Learning to Rank using Gradient Descent.** *ICML 2005*

[14] **LambdaRank**

* source code :[[code]](https://github.com/ChenglongChen/tensorflow-LTR) [[code]](https://github.com/Jabberwockleo/lambdarank) 
* paper :[[pdf]](https://pdfs.semanticscholar.org/fc9a/e09f9ced555558fdf1e997c0a5411fb51f15.pdf)  

**Learning to Rank with Nonsmooth Cost Functions.** *NIPS 2006*

[15] **LambdaMART** 

* source code :[[code]]() 
* paper :[[pdf]](https://www.microsoft.com/en-us/research/publication/adapting-boosting-for-information-retrieval-measures/?from=http%3A%2F%2Fresearch.microsoft.com%2Fpubs%2F102750%2Flambdamart_final.pdf) 

**Adapting Bboosting for Information Retrieval Measures**

[16] **RankNet/LambdaRank/LambdaMART:turtorial**

* paper :[[pdf]](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/) 
* review : [[post]](https://medium.com/@nikhilbd/intuitive-explanation-of-learning-to-rank-and-ranknet-lambdarank-and-lambdamart-fe1e17fac418)

**From RankNet to LambdaRank to LambdaMART: An Overview**

[17] **TF-Ranking** 

* source code :[[code]](https://github.com/tensorflow/ranking) 
* paper :[[pdf]](https://arxiv.org/abs/1812.00073) google
* review :[[review]](https://ai.googleblog.com/2018/12/tf-ranking-scalable-tensorflow-library.html) [[review]](https://github.com/chullhwan-song/Reading-Paper/issues/163)

**TF-Ranking: A Scalable TensorFlow Library for Learning-to-Rank.** *2018*

> TF ranking open-source


[18] **FNRM** 

* source code :[[code]](https://github.com/mikvrax/TrecingLab) 
* paper :[[pdf]](https://arxiv.org/abs/1704.08803)
* review : [[review]](https://github.com/rejasupotaro/paper-reading/issues/15) [[review]](https://mostafadehghani.com/2017/04/23/beating-the-teacher-neural-ranking-models-with-weak-supervision/) [[review]](https://mostafadehghani.com/wp-content/uploads/2016/07/SIGIR2017_Presentation.pdf)

**Neural Ranking Models with Weak Supervision.** *CoRR 2017*

[19] **PACRR**  

* source code :[[code]](https://github.com/nlpaueb/deep-relevance-ranking)  [[code]](https://github.com/MatanRad/Neural-IR-Project) 
* paper :[[pdf]](https://arxiv.org/abs/1704.03940)
* review : [[review]](https://khui.github.io/slides/pacrr-emnlp17.pdf)

**PACRR: A position-aware neural IR model for relevance matching.** *EMNLP 2017*

[20] **DRMM & PACRR** 

* source code :[[code]](https://github.com/nlpaueb/deep-relevance-ranking)  
* paper :[[pdf]](https://arxiv.org/abs/1704.08803) google
* review : [[review]](http://nlp.cs.aueb.gr/pubs/EMNLP2018Preso.pdf)

**Deep Relevance Ranking Using Enhanced Document-Query Interactions.** *EMNLP 2018*

[21] **turtorial**  

* paper :[[pdf]](https://www.microsoft.com/en-us/research/publication/introduction-neural-information-retrieval/) MS

**An Introduction to Neural Information Retrieval**

[22] **rank-text-cnn**  

* source code :[[code]](https://github.com/gvishal/rank_text_cnn) [[code]](https://github.com/zhangzibin/PairCNN-Ranking)
* paper :[[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.723.6492&rep=rep1&type=pdf)

**Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks.** *SIGIR 2015* 

* source code :[[code]](https://github.com/zhangzibin/PairCNN-Ranking)

[23] **ConvRankNet** 

* source code :[[code]]() 
* paper :[[pdf]](https://arxiv.org/abs/1802.08988)
* review :[[review]](https://github.com/chullhwan-song/Reading-Paper/issues/256)

**Deep Neural Network for Learning to Rank Query-Text Pairs.** *2018*

[24] **Listwise Neural Ranking Models** 

* source code :[[code]]()
* paper :[[pdf]](https://dl.acm.org/citation.cfm?id=3341981.3344245)

**Listwise Neural Ranking Models.** *ICTIR 2019*


[25] **DLCM** 

* source code :[[code]](https://github.com/QingyaoAi/Deep-Listwise-Context-Model-for-Ranking-Refinement) 
* paper :[[pdf]](https://arxiv.org/abs/1804.05936)

**Learning a Deep Listwise Context Model for Ranking Refinement.** *SIGIR 2018*


### Join BERT

[26] **CEDR** : [2019.12.18 Current SOTA](https://paperswithcode.com/sota/ad-hoc-information-retrieval-on-trec-robust04)

* source code :[[code]](https://paperswithcode.com/paper/190407094)
* paper :[[pdf]](https://arxiv.org/abs/1904.070946)
* review :[[review]](https://github.com/chullhwan-song/Reading-Paper/issues/262)

**CEDR: Contextualized Embeddings for Document Ranking**


[27] **BERT-MaxP**

* source code :[[code]](https://paperswithcode.com/paper/deeper-text-understanding-for-ir-with)
* paper :[[pdf]](https://arxiv.org/abs/1905.09217v1)

**Deeper Text Understanding for IR with Contextual Neural Language Modeling**


[28] **NPRF**

* source code :[[code]](https://github.com/ucasir/NPRF)
* paper :[[pdf]](https://arxiv.org/pdf/1810.12936v1.pdf)

**NPRF: A Neural Pseudo Relevance Feedback Framework for Ad-hoc Information Retrieval**

[29] **BERT FT**

* source code :[[code]](https://github.com/castorini/birch)
* paper :[[pdf]](https://arxiv.org/pdf/1903.10972v1.pdf)

**Simple Applications of BERT for Ad Hoc Document Retrieval**

[30] **GSF**

* source code :[[code]](https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/python/model.py)
* paper :[[pdf]](https://research.google/pubs/pub48348/)

**Learning Groupwise Multivariate Scoring Functions Using Deep Neural Networks**


