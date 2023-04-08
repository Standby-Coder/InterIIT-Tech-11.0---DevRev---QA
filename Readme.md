
# DevRev : HighPrep

## Expert Answers in a Flash: Improving Domain-Specific QA

  

This project was a submission for one of the high preparation problem statements offered at Inter IIT Tech Meet 11.0 hosted by IIT Kanpur.

The details of the problem statement can be found here: ()

  

## Team Members

- Kushagra Bhushan

- Neeraj Anand

- Nahush Lele

- Saksham Dwivedi

- Saswati Subhalaxmi

- Keshav Reddy

- M. Thirulok Sundar

- Subhrajit Mukherjee

- Sohith Bandari

- Aryamaan Thakur

  

## Introduction
### Task Description
We would be given a question and a set of paragraphs. From these our model would predict if the question can be answered with the given paragraphs. Each question and paragraph is associated with a specific theme. If answer is possible our model will also predict the exact answer from the predicted paragraph by predicting the start_index and the answer_text field for the given question.
### Constraints
**Training**
- Teams are allowed to train their model whichever way they want but the same must be reproducible while running your training colab notebook on a system with specifics similar to free tier google collab system with:-
-- No hardware accelerator
--12 GB System RAM
-- Within 12 hrs
- Pre-trained model
-- Open source model published before 1 Dec’22
--This needs to be mentioned in the end term report with valid reference.

**Testing**
- Inference notebook must run on a system with specifics similar to free tier google collab system with:-
--No hardware accelerator
-- 12 GB System RAM
###  Dataset Description
Training dataset
The dataset contains the following fields:
1. Question: Question for which answer is to be found
2. Theme: Name of the domain this question & paragraph belongs to. For e.g. a paragraph could be on the theme “cricket” / “mathematics” / “biology” etc.
3. Paragraph: Paragraph from the mentioned theme which may contain the answer
4. Answer_possible: If the answer is possible from the given paragraph
5. Answer_text: Answers from the given paragraph
6. Answer_start: Index position from where the answer starts
Training dataset is available here:
https://drive.google.com/file/d/1Zyb752A3o7b9dqrGt24XU0sl53FVqya/view?usp=share_link
### Evaluation Metrics
Accuracy metric for paragraph prediction:
- True positive: If the predicted paragraph exists in the ground truth list of paragraphs which can answer the query.
- True negative: If predicted that there does not exist a paragraph which can answer the query and that indeed is the case.
- Instead of F1(as originally mentioned in PS), we’ll be evaluating the accuracy metric:
- Accuracy: (True positive + True negative) / (Total number of queries)

F1 score for QA task:
- For a given query, assume there are 3 answers in ground truth: “random token word”, “token word problem”, “word problem pushed”.
- For a predicted answer, “problem pushed”, it’ll calculate the maximum F1 score while comparing it with all the 3 possible answers.
- In the above example, max F1 score would be ⅘ and the same would be taken in account for this query.
- Final score for a theme would be avg. F1 score over all queries in that theme.

Inference time:
- Metric score for a theme would be F1 score Q/A task + Accuracy for paragraph prediction
- If our average inference time(AIT) for a theme is greater than 200 ms then:
Final score for theme = (200/AIT(ms)) * Metric score for theme

Final Score
- Final score = Σ theme_weight * (final score for that theme)
- Theme weight would not be exposed to teams.
## Approaches

  

- **Using Retrospective Reader:**

We started off with researching about the task at hand, we found that the Stanford Question Answering Dataset (SQuADv2) dataset is widely used as a benchmark for Question-Answering tasks and is quite similar to the given data. We found a research paper that had a high F1 score on SQuADv2, Retrospective Reader for Machine Reading Comprehension (Zhang et al. 2021). We implemented the method and had promising results on the given training dataset with an F1 score of around 0.89 using ELECTRA as the backbone. Retro Reader works well because it imitates the human way of text comprehension, by roughly looking for answers in the given passage and then actually finding out the answer. The issue with this method was it was very hardware intensive requiring 4 hours to train on Colab Notebook.

<img  src="https://media.arxiv-vanity.com/render-output/6106993/x1.png">

- **Classical ML methods with lower computational budget training**

We moved on to using Bi-directional LSTMs taking inspiration from a master’s project from Santa Carla University. The method involved training two bidirectional LSTMs for extraction of embeddings and then computing a cosine similarity between them to ascertain whether the question is answerable or not. We tried training this model using only the given constraints i.e. CPU but training took over 30 hours. We then proceeded to use GPU to train it to completion but the F1 score was far too low to be considered. Thus we had to give this method up as well.

- **Using Pretrained Models on SQuADv2 dataset**

Our current method is to use a **model pretrained on the SQuADv2 dataset**. This dataset is very similar to the given training set, in that it has questions which have to be answered given a context. Considering that we have **significantly lesser data** than in SQuADv2 and the **hardware constraints**, we decided not to fine-tune the pretrained model on our dataset. On evaluating 8 backbones trained on SQuADv2 **using the metric that we will be judged on** we decided to use tiny- RoBERTa as the final model to be used. We found that larger models had a **much larger inference time** and although had better F1 scores their average inference times offset their final metric score so that they would perform worse. This is why we only tested light weight models for inference of the test set. We created the test set using **Group K-Fold** method which is further explained in the next section. Our experimental results using the 8 backbones were as follows: -

{https://drive.google.com/file/d/1m7dr9ZjhAUDExziFGsIJHSKxZQXILQyy/view?usp=share_link}


As we can see RoBERTa tiny model has a very good para and qa score because of its quick inference time. RoBERTa was originally trained for being more robust than BERT and the tiny model refers to a smaller version of the original model.

**Data Split Generation: -**

Initially when we planned to train our own model, we split the dataset provided to us to perform validation. As one of the evaluation tasks (Task 1) was going to test our model to generalise to unseen themes we figured a good way to set up data splits would be using Group K-Fold. In this method, the data is divided in such a way that always a certain class of the target variable is unavailable in the training set. This is done to validate the generalisability of the model. When we realised we couldn’t train our own model, we reused the splits to evaluate the pre-trained models to get the best one.
<img  src="https://amueller.github.io/aml/_images/kfold_cv.png">
## Our Final Model
{https://drive.google.com/file/d/1c86wFokZpiLEA4_aHejpoV6lytVbn36z/view?usp=share_link}

# Optimizations

Since the metric penalizes latency and we had some hardware constraints(no GPUS to be used), we tried different ways to optimize inference time.
##  Choice of models
We used only lightweight models since they had a great inference time advantage which compensated for the slightly poorer performance compared to larger models. The models we tried were:

 - MiniLM
 - Tiny BERT  
 -  RoBERTa Tiny  
 -  MobileBERT  
 -  DistilBERT

## Precomputing context embeddings

We pre-computed the context embeddings so that we did not have to do this while inference. The only  
embeddings calculated during the inference time were the new questions embedding.  Doing this reduced our total inference time by about 5 seconds.

##  Parallelizing on the CPU

●  We used Ray Library to speed up our inference time. This allowed us to exploit the multiple cores of the CPU to parallelize and create n instances of our model (where n = number of CPU cores).
●  Doing this reduced our average inference time by a factor of n. This is because the  
models were running in parallel. We worked on Google Colab which provides 2 cores. Thus this halved our inference time.

# Final Result and Conclusion 
Our proposed pipeline achieves a final metric of 0.74 for paragraph retrieval task and 0.71 for  
question-answering task.We believe it can be further improved using a little more data and better hyper parameter tuning as mentioned in our proposed experiments.  
The major drawback we had other than the hardware constraints was the lack of data and inability to use external data(external data was not allowed). This affected our results greatly.

## Run time analysis of the Contrastive Learning Models
●  Training Time for Each Theme:  1 minute  
●  Inference Time:  0.001 seconds for 1 embedding  
●  Model Size:  12 Mb

## Literature Review
1)  Unsupervised Document Embedding via Contrastive Augmentation by  Luo et al.
      https://arxiv.org/pdf/2103.14542.pdf 
2)  Adaptive Margin Diversity Regularizer for handling Data Imbalance in Zero-Shot  
     SBIR by  Dutta et al.  
     https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500341.pdf
3)  Towards Universal Paraphrastic Sentence Embeddings by  Wieting et al.  
     https://arxiv.org/abs/1511.08198
4)  SimCSE: Simple Contrastive Learning of Sentence Embeddings by  Gao et al.
     https://arxiv.org/abs/2104.08821
## Further Work
There are a few possible ways  to further improve the quality of the retrieved documents. We list a few here:

-   Averaged Query Expansion:
    -   Use the top-k retrieved document and average their embeddings
    -   Forms a query from within the distribution of the contexts.
    -   Re-retrieve the top-k passages using this query
-   Increase per context query size.
-   Work on improving document embeddings i.e. use the increased dataset to train models which use methods like Adaptive Margin Regularization.
