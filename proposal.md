# 11-785: Early Pancreatic Cancer Detection

|Joshmin Ray
Department of Machine Learning
Carnegie Mellon University
Pittsburgh, PA 15213
joshminr@andrew.cmu.edu|
|Rohan Chawla
Department of Computer Science
Carnegie Mellon University
Pittsburgh, PA 15213
rchawla2@andrew.cmu.edu|
|Jesse Shen
Department of Biomedical Engineering
Carnegie Mellon University
Pittsburgh, PA 15213
jwshen@andrew.cmu.edu|
|Logan Nye, MD
Department of Computer Science
Carnegie Mellon University
Pittsburgh, PA 15213
lnye@andrew.cmu.edu|
From our proposal:
## 1 Introduction
Pancreatic cancer is an aggressive form of cancer that is typically detected in late stage with a low
long-term survival rate, ranging from 2-9% at 5 years Davide Placido [2023]. It presents as a leading
cause of death among cancers in the United States and is projected to continue impacting lives in
the coming decades Rahib et al. [2014], with a projected increase in incidence rate by 2030. Major
research institutions such as the U.S. National Institute of Health (NIH) attribute much of pancreatic
cancer’s poor prognosis from late detection that leaves limited and ineffective options for treatment
Ramírez-Maldonado et al. [2024]. Some of the reasons for this late stage detection include the
inability to adequately visualize the pancreas given its retro-peritoneal location, its lack of specific
molecular markers, and its indistinct early clinical symptoms Tripathi et al. [2024]. One approach to
early detection is artificial intelligence.
According to Patel et al. [2024], there has been an exponential increase in the number of published
research regarding AI and the pancreas. One of the approaches we found in the literature is the
application of deep learning algorithms to predict the risk of pancreatic cancer. This will lead to
possible early detection.
Our research study aims to build upon an existing deep learning algorithm for pancreatic cancer
risk prediction reported in Nature Medicine in March 2023. We model our methodology as outlined
in the baseline study as a foundation. Our goal is to enhance the predictive accuracy and clinical
applicability of this AI model for pancreatic cancer prediction by leveraging the rich, multimodal
patient data available in the MIMIC-IV and MIMIC-IV-Note datasets. These enhancements are
intended to address potential limitations of the baseline study and incorporate alternative AI modeling
techniques across diverse data types to improve early detection and personalized risk assessment of
pancreatic cancer.
## 2 Literature Review
The field of AI-related early cancer detection uses multiple means to calculate a patient’s risk. These
methods take in input data from one of the following modalities: patient records, medical images, or
biomarkers. With pancreatic cancer (PC), all three means have been studied as promising diagnostic
indicators Tripathi et al. [2024]. In this section, we will cover these three lines of research as well as
some research with the specific patient record dataset, Medical Information Mart for Intensive Care
(MIMIC) - IV, we intend to use.
Biomarker-based classification offers precise detection only possible at the molecular level. The
diagnostic field with most relevance to AI is liquid biopsy research, which seeks to identify markers
for cancer via a sample of non-solid biological tissue– usually blood. Known biomarkers can be
proteins, circulating DNA/RNA sequences, and circulating tumor cells Ramírez-Maldonado et al.
[2024]. Current machine learning (ML) models applying biomarker detection exclusively analyze
genetic or protein profiles of patients Patel et al. [2024]. At the same time, there remains active debate
regarding biomarkers themselves. A given biomarker’s predictive value needs to be generalized and
effectively biochemically isolating and detecting molecules of interest continue to be active areas of
research Ramírez-Maldonado et al. [2024].
Applying deep learning models to medical imaging data shows promise in both early diagnosis
and characterizing subtypes of cancers for treatment. PC is a solid tissue tumor that is identified
when patients report symptoms via a computed tomography (CT) scan. Other imaging modalities
used with AI in pancreatic cancer diagnosis include magnetic resonance imaging (MRI) and EUS
(endoscopic ultrasound). The field of radiomics specifically seeks to identify the minute details of
medical images identified by machine learning models to aid medical professionals with models
further able to classify cancers for optimal treatment. With an emphasis on early diagnosis for PC,
AI algorithms have demonstrated high sensitivity, but pancreatic tumors with a spherical diameter
of less than 2 cm prove difficult to detect Alexandra Corina Faur [2023]. While imaging remains
recommended for high-risk individuals by medical professionals, this preferred method is currently
impractical for the general population Klein [2021]
A "top-down" approach comes from analyzing patient data. The earlier the diagnosis the better the
prognosis, which is the impetus of deep learning models analyzing large databases of patient records
Muhammad et al. [2019]. Tracing the cause of a given cancer has also proven useful in characterizing
it and providing targeted treatment Hu et al. [2019]. Diagnostic tests are often focused on high-risk
individuals, leaving the potential for large patient record databases to detect PC early in the general
population. However, the particular risk factors of interest are not clear for early PC diagnosis. Deep
learning models are being researched as a solution, in conjunction with large patient databases used
to study patient trajectory Davide Placido [2023].
One of the most common applications of the MIMIC-IV’s patient data so far has been determining
diseases diagnoses from blood-glucose data. One paper identified type 2 diabetes cases and another
used the variability in blood-glucose to detect coronary artery disease Zhang W [2023], He et al.
[2024]. These examples may be additional variables to consider for our project, besides the other
patient data used, since diabetes emergence is a risk factor for PC Klein [2021].

## 3 Model Description
For our baseline, we chose to use a transformer model from a recent paper for predicting pancreatic
cancer Davide Placido [2023]. Their model treats patients as time series data. Each patient’s hospital
record is compressed down to a series of disease code diagnoses. After each hospital visit, a diagnosis
for the patient is made, which is represented as a code using the International Classification of Disease
(ICD) standard. Each of these codes is associated with a time stamp of when the diagnosis was made.
The authors used a subset of these codes, level 3 codes, which were most relevant for pancreatic
cancer diagnoses. They also used positional embeddings to introduce the timing of these diagnoses
into the vector representations of the series of codes. The positional embeddings are dependent on
the age of the patient at the time of the diagnosis and the time difference between the diagnoses.
After the embedding is complete, the transformer is used to create attended vectors for each patient,
and then fed into an MLP to generate the final output. This output is a set of 5 probabilities, for the
likelihood of cancer occurrence at 5 different time points after the final diagnosis: 3 months after, 6
months after, 12 months after, 36 months after and 60 months after. The loss function used is a cross
entropy function which compares each of these probabilities to the true cancer diagnosis values (0
before diagnosis and 1 after). They also used L2 regularization. We tested this baseline against our
planned dataset, MIMIC-IV, as we could not access the dataset from the original paper.
The authors used the area under the receiving operator characteristic (AUROC) as their performance
metric and obtained the following performance for their dataset(AUROC = 0.879 (0.877-0.880))
Davide Placido [2023].
## 4 Baseline Selection
There have been several different approaches to using machine learning in the field of cancer detection,
ranging from support vector machines (SVMs) and random forests to deep neural networks, natural
language processing (NLP), and transformers Patel et al. [2024]. There have also been both text and
imaging approaches.
The paper we chose as a baseline compares sequential neural networks, gated recurrent unit (GRU)
models, and the Transformer model. These were compared against their baseline bag-of-words model,
which used disease codes as the words and ignores time and order of disease events Davide Placido
[2023]. They found the transformer to work the best. This paper was one of few in the field to use
time series data and deep learning to tackle this problem, which was why it was appealing to us. The
Mamba architecture serves as a more efficient alternative to transformer, so finding a transformer
baseline is very helpful in guiding how to structure our data. This baseline also reveals how limited
the transformer model is for this task. Each patient can only be treated as a series of diagnostic codes,
while clinical data contains a much greater wealth of information than just these codes. We plan to
use the MIMIC-IV dataset, which includes many different types of data. This includes text data in
the form of clinical notes, as well as lab results, monitoring data and diagnoses. Ideally, we hope
to incorporate all of these types of data into the model, for better generality. One major issue with
the transformer model is that its parameter size scales quadratically with the length of the sequence,
so training a reasonably sized model requires limiting the length of the sequence. Using a Mamba
model would allow us to work with longer sequences, and thus incorporate more types of data per
patient. Another important aspect of this model is that it performs well at predicting cancer risk at
several defined periods of time. This can be a more useful output for doctors than just a single risk
score, which is difficult to confirm or act upon.

References
Jessica X. Hjaltelin Chunlei Zheng Amalie D. Haue Piotr J. Chmura Chen Yuan Jihye Kim Renato Umeton
Gregory Antell Alexander Chowdhury Alexandra Franz Lauren Brais Elizabeth Andrews Debora S. Marks
Aviv Regev Siamack Ayandeh Mary T. Brophy Nhan V. Do Peter Kraft Brian M. Wolpin Michael H. Rosenthal
Nathanael R. Fillmore Søren Brunak Chris Sander Davide Placido, Bo Yuan. A deep learning algorithm to
predict risk of pancreatic cancer from disease trajectories. Nature Medicine, 2023.
Lola Rahib, Benjamin D. Smith, Rhonda Aizenberg, Allison B. Rosenzweig, Julie M. Fleshman, and Lynn M.
Matrisian. Projecting cancer incidence and deaths to 2030: The unexpected burden of thyroid, liver, and
pancreas cancers in the united states. Cancer Research, 05 2014. doi: https://doi.org/10.1158/0008-5472.
CAN-14-0155.
Elena Ramírez-Maldonado, Sandra López Gordo, Rui Pedro Major Branco, Mihai-Calin Pavel, Laia Estalella,
Erik Llàcer-Millán, María Alejandra Guerrero, Estrella López-Gordo, Robert Memba, and Rosa Jorba.
Clinical application of liquid biopsy in pancreatic cancer: A narrative review. International Journal of
Molecular Sciences, 2024. doi: https://doi.org/10.3390/ijms25031640.
Satvik Tripathi, Azadeh Tabari, Arian Mansur, Harika Dabbara, Christopher P. Bridge, and Dania Daye. From
machine learning to patient outcomes: A comprehensive review of ai in pancreatic cancer. Diagnostics, 14(2),
2024. ISSN 2075-4418. doi: 10.3390/diagnostics14020174. URL https://www.mdpi.com/2075-4418/
14/2/174.
3
Hardik Patel, Theodoros Zanos, and D. Brock Hewitt. Deep learning applications in pancreatic cancer. Cancers,
16(2), 2024. ISSN 2072-6694. doi: 10.3390/cancers16020436. URL https://www.mdpi.com/2072-6694/
16/2/436.
Laura Andreea Ghenciu Alexandra Corina Faur, Daniela Cornelia Lazar. Artificial intelligence as a noninvasive
tool for pancreatic cancer prediction and diagnosis. World Journal of Gastroenterology, 2023. doi: DOI:
10.3748/wjg.v29.i12.1811.
Alison P. Klein. Pancreatic cancer epidemiology: understanding the role of lifestyle and inherited risk factors.
Nature Reviews Gastroenterology Hepatology, 2021. doi: https://doi.org/10.1038/s41575-021-00457-x.
Wazir Muhammad, Gregory R. Hart, Bradley Nartowt, James J. Farrell, Kimberly Johung, Ying Liang, and Jun
Deng. Pancreatic cancer prediction through an artificial neural network. Frontiers in Artificial Intelligence, 2,
2019. ISSN 2624-8212. doi: 10.3389/frai.2019.00002. URL https://www.frontiersin.org/articles/
10.3389/frai.2019.00002.
Jessica X. Hu, Marie Helleberg, Anders B. Jensen, Søren Brunak, and Jens Lundgren. A Large-Cohort,
Longitudinal Study Determines Precancer Disease Routes across Different Cancer Types. Cancer Research,
79(4):864–872, 02 2019. ISSN 0008-5472. doi: 10.1158/0008-5472.CAN-18-1677. URL https://doi.
org/10.1158/0008-5472.CAN-18-1677.
Cao T. Zhang W. Automated type 2 diabetes case and control identification from the mimic-iv database. AMIA
Jt Summits Transl Sci Proc., 2023.
Hao-Ming He, Shu-Wen Zheng, and et al. Simultaneous assessment of stress hyperglycemia ratio and glycemic
variability to predict mortality in patients with coronary artery disease: a retrospective cohort study from the
mimic-iv database. Cardiovascular diabetology, 2024. doi: https://doi.org/10.1186/s12933-024-02146-w.
Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. 2023.
4

