# **Reproduction**

**Jack Casey** *23357614*
**Jean Carson** *23303972*
**Ushen Wijayaratne** 23362073

A report in fulfilment of the CS4445: Artificial Intelligence Topic
Spring Semester Team-Based Project

**![1-UL\_Logo\_CMYK][image1]**

University of Limerick

Immersive Software Engineering

March 2025

**Table of Contents**

**[1\. Abstract	3](#abstract)**

[**2\. Introduction	3**](#introduction)

[2.1 Project motivation and objectives	3](#2.1-project-motivation-and-objectives)

[2.2 Project Team Structure and Division of Labor	4](#2.2-project-team-structure-and-division-of-labor)

[**3\. Dataset	5**](#dataset)

[**4\. Exploratory Data Analysis	6**](#exploratory-data-analysis)

[4.1 Extracting Demographic Data	8](#4.1-extracting-demographic-data)

[**5\. Preprocessing	10**](#preprocessing)

[5.1 Preprocessing Goal	10](#5.1-preprocessing-goal)

[5.2 Creating Trios	11](#5.2-creating-trios)

[5.3 Generating Keypoints	14](#5.3-generating-keypoints)

[5.4 Calculating Face Angles	16](#5.4-calculating-face-angles)

[5.5 Challenges with 2D Pose Estimation	19](#5.5-challenges-with-2d-pose-estimation)

[5.6 Challenges upscaling images	20](#5.6-challenges-upscaling-images)

[5.7 Aligning and Blurring the Dataset	24](#5.7-aligning-and-blurring-the-dataset)

[5.8 Converting into format for Latent Vector Encoding	25](#5.8-converting-into-format-for-latent-vector-encoding)

[**6\. Experimentation	27**](#experimentation)

[6.1 What didn’t work?	27](#6.1-what-didn’t-work?)

[6.2 Metrics	29](#6.2-metrics)

[6.3 Results	29](#6.3-results)

[Observations:	29](#observations:)

[**7\. ML Model / Network Structure	33**](#ml-model-/-network-structure)

[7.1 StyleGAN Architecture and Implementation	33](#7.1-stylegan-architecture-and-implementation)

[7.1.1 Model Selection	33](#7.1.1-model-selection)

[7.1.2 Encoder Selection	34](#7.1.2-encoder-selection)

[7.2 Proposed Kinship Synthesis Model	34](#7.2-proposed-kinship-synthesis-model)

[7.2.1 Parent Image Embedding	34](#7.2.1-parent-image-embedding)

[7.2.2 Latent Space Blending	34](#7.2.2-latent-space-blending)

[7.2.2.1 Simple Weighted Average	35](#7.2.2.1-simple-weighted-average)

[7.2.2.2 Per-Dimension Weighting	36](#7.2.2.2-per-dimension-weighting)

[7.2.2.3 Layer-Wise Weighting	36](#7.2.2.3-layer-wise-weighting)

[7.2.2.4 Binary Mask Blending	36](#7.2.2.4-binary-mask-blending)

[7.2.3 Feature Refinement with InterfaceGAN	37](#7.2.3-feature-refinement-with-interfacegan)

[**7.3 Neural Optimization Framework for Kinship Face Generation	37**](#7.3-neural-optimization-framework-for-kinship-face-generation)

[7.3.1 Overview	37](#7.3.1-overview)

[7.3.2 Architecture and Implementation	37](#7.3.2-architecture-and-implementation)

[7.3.2.1 Adaptive Dimension Weight Model	37](#7.3.2.1-adaptive-dimension-weight-model)

[Model Explanation	39](#model-explanation)

[7.3.2.2 Differentiable Training Pipeline	39](#7.3.2.2-differentiable-training-pipeline)

[7.4 Training Methodology	41](#7.4-training-methodology)

[7.4.1 Loss Function	41](#7.4.1-loss-function)

[7.4.2 Training Process	42](#7.4.2-training-process)

[Code Explanation	43](#code-explanation)

[7.4.3 Latent Combination Mechanism	44](#7.4.3-latent-combination-mechanism)

[Code Explanation	44](#code-explanation-1)

[7.4.4. Overfitting and attempts to combat it	45](#7.4.4.-overfitting-and-attempts-to-combat-it)

[8\. Results and Evaluation	47](#results-and-evaluation)

[8.1 Data	47](#8.1-data)

[8.2 Challenges in Accuracy Assessment	49](#8.2-challenges-in-accuracy-assessment)

[8.3 Technical Challenges and Memory Optimization	50](#8.3-technical-challenges-and-memory-optimization)

[8.3.1 Memory Requirements and Initial Failures	50](#8.3.1-memory-requirements-and-initial-failures)

[8.3.2 Memory Reduction Techniques	50](#8.3.2-memory-reduction-techniques)

[Mixed Precision Training/Inference	50](#mixed-precision-training/inference)

[Memory-Efficient Model Loading	51](#memory-efficient-model-loading)

[Explicit Memory Management	51](#explicit-memory-management)

[8.3.3 Results of Memory Optimization	52](#8.3.3-results-of-memory-optimization)

[**9\. Experiments & Evaluation	53**](#experiments-&-evaluation)

[Fairness & Bias Considerations	53](#fairness-&-bias-considerations)

[**10\. HCI \- A web app	54**](#hci---a-web-app)

[**11\. Discussion and Conclusion	55**](#discussion-and-conclusion)

[11.1 Key Findings	55](#11.1-key-findings)

[11.2 Technical Challenges	56](#11.2-technical-challenges)

[11.3 Future Work	56](#11.3-future-work)

[11.4 Conclusion	57](#11.4-conclusion)

[**12\. References	58**](#references)

##

##

1. ## **Abstract** {#abstract}

Our project focuses on kinship-based facial generation, aiming to create a realistic visual representation of a hypothetical child from two parent images. Utilising machine learning techniques, we used StyleGAN2’s W+ latent space for feature blending and InterfaceGAN for targeted attribute manipulation. Our approach combines parent image embeddings through various blending strategies, including simple weighted averaging, per-dimension weighting, and layer-wise weighting. Through extensive experimentation, we optimized blending methods to achieve more realistic and coherent child images. Key challenges included memory limitations during training and ensuring feature consistency, which we mitigated using memory-efficient techniques and explicit GPU memory management. This project has potential applications in forensic age progression, historical lineage visualization, and personalized portrait generation. Future work will focus on improving feature interpretability and refining the neural optimization framework to enhance the quality and diversity of generated images.

2. ## **Introduction** {#introduction}

### 2.1 Project motivation and objectives {#2.1-project-motivation-and-objectives}

Our project aims to develop an interface that allows prospective parents to visualize a hypothetical child based on their photos. By uploading images of both parents, the system will generate a predicted child’s face using advanced machine learning techniques.

Beyond being an engaging demonstration of cutting-edge technology, this project has practical real-world applications, such as:

* **Missing Person Aging Simulation** – Our model can be trained to manipulate facial attributes in a controlled manner, allowing for age progression and other modifications. This capability could assist in generating aged-up images of missing individuals, improving search and rescue efforts.
* **Historical and Ancestral Reconstruction** – The tool could be used to approximate how historical figures' descendants might have looked or to reconstruct family lineage visuals.
* **Personalized Art & Portrait Generation** – Users could create artistic renderings of hypothetical future children for fun or as sentimental keepsakes.

By combining innovation with meaningful real-world uses, our project explores the potential of AI-driven facial prediction while offering an interactive and insightful experience for users.

### 2.2 Project Team Structure and Division of Labor {#2.2-project-team-structure-and-division-of-labor}

Our team approached this project through collaborative planning followed by specialized implementation roles. We began with extensive group discussions to define requirements, evaluate approaches, and make key architectural decisions. After careful consideration, we decided not to implement a GAN from scratch, but instead to train a model that would create optimal child faces using a pre-trained GAN. This strategy allowed us to focus our efforts on the novel aspects of our system while leveraging established generative models.

Once we established the overall approach, we divided the implementation based on each member's technical strengths:

* **Ushen: Data Preprocessing** \- Responsible for developing the face detection and alignment pipeline and creating preprocessing workflows for normalizing input images. This included configuring the face landmark detection system and ensuring consistent image formatting across the pipeline.

* **Jack: Method 1 (Neural Optimization)** \- Led the development of our primary approach using neural optimization for latent space blending. Implementing alternative blending methods (uniform and custom weights) for comparison with the neural approach. This involved designing and implementing the AdaptiveDimensionWeightModel architecture, creating differentiable training pathways.

* **Jean: Method 2 (Alternative Approaches & Interface)** \- Developing both the web application interface and command-line tools. Jean also led work on the experiment that ultimately was not integrated, investing significant time exploring an approach that proved unviable but provided valuable insights for the successful methods. This comprehensive work included implementing the Flask-based user interface, visualization utilities, and documenting lessons learned from the unsuccessful experiment.

Throughout the project, we maintained regular communication through weekly meetings and a shared GitHub repository. While each member had primary responsibilities, we collaborated closely on system integration, testing, and evaluation. This division of labor allowed us to address the complex technical challenges of the project efficiently while ensuring cohesive integration of all components.

3. ## **Dataset** {#dataset}

After researching other StyleGan projects and considering datasets such as *TSKinface* and *KinFaceW,*  we settled on the *Family101* dataset. This dataset is large, containing 101 multi generational families. There are 206 nuclear families with 607 individuals and 14,816 images. *KinFaceW* is equally large but focuses more on 1:1 kinship (eg mother-daughter, father-son), and would require significant work to even determine if we could establish family triplets (mother, father, child) without offering advantages over *Family101. TSKinFace* did offer family triplets however the dataset was far less racially diverse, without compensating with an advantage on size, image alignment or organisation. Therefore, we chose to work with *Family101* and studied its contents in further depth.

In this dataset, each family includes 1 to 7 atomic families. Citing the paper where the dataset was created, it consists of 72% Caucasians, 23% Asians and 5% African Americans. It also excludes non-biologically related parents and children. (*Fang, R et. al, 2013*)

It also comes with a text file outlining how the family members are mapped together. The first column is a serial number 0, which indicates the start of a new extended family, while 1, 2, etc., is a sequence of nuclear families in this extended family. The second column covers the roles in a family such as husband, wife, daughter and son. The third column is the full name of the individual.

![][image2]

*Fig 3.1 Example snippet from family structure file*

4. ## **Exploratory Data Analysis** {#exploratory-data-analysis}

Analysing the dataset, we confirmed that there were 101 unique extended families. However, there were a few discrepancies with 209 nuclear families and 633 unique individuals compared to the 206 and 607 noted in the paper. We found a few errors where the numbering on the nuclear families was off, such as how there are two nuclear families for Presley labelled 1\.

![][image3]

*Fig 4.1 Family structure file*

Looking at the data distribution, we noticed there was a slight skew towards males over females. There is an average of around 7 members in an extended family.

![][image4]

*Fig 4.2 Male/female database split*

![][image5]

*Fig 4.3 Distribution of family roles*

![][image6]

*Fig 4.4 Distribution of extended family sizes*

Looking at missing values, 54 nuclear families are incomplete.

![][image7]

*Fig 4.5 Incomplete family exploration*

Therefore, the total number of completed nuclear families is 155\.

### 4.1 Extracting Demographic Data  {#4.1-extracting-demographic-data}

We used Deepface to explore the demographic distribution of our dataset. We iterated through the images and extracted features using pre-trained models like VGG-Face and Google FaceNet to create feature maps. These feature maps are passed into separate classification models that give us age, gender, race and emotion data.

![][image8]

![][image9]
*Fig 4.6 Visualisations of the Demographic CSV using Seaborn and Pyplot*

5. ## **Preprocessing** {#preprocessing}

### 5.1 Preprocessing Goal {#5.1-preprocessing-goal}

After dropping all the incomplete nuclear families. We began establishing preprocessing goals. We settled on having a function that would return three lists of images: mothers, fathers and children. Whereby mothers, fathers and children would all belong to the same trio.

The images themselves would have to be square, having equal height and width, and the resolutions have to be powers of 2\.

Resolutions being a power of 2 is a common practice in the architecture of Generative Adversarial Networks like StyleGan. This is due to how operations like pooling, striding and upsampling are structured. Pooling and striding reduce the spatial dimensions of feature maps by factors of 2\. Using input resolutions of powers of 2 ensures these dimensions can be repeatedly halved until a size of 1x1 is reached. Similarly, generators in GANs often use upsampling layers with factors of 2 to progressively increase the resolution of the generated image.

With Stylegan2 specifically, while the ideal input resolution is 1024x1024, the primary input resolution mentioned for general use according to () is 256x256.

Additionally, the images would have to be upscaled from the current **120x150** pixel, aligned and normalised.

###

### 5.2 Creating Trios {#5.2-creating-trios}

We began formulating the data into trios: a mum, a dad and a child. It would be in the form of unique combinations, eg, if parent A and parent B have two kids, B and C. The trio would be A, B, C and A, B, D. Each trio would have a list of images associated with it.

We conducted data visualisation to understand the spread of trios per extended family. The largest is the Kennedy family, with 18 trios in the extended family.

![][image10]

*Fig 5.2.1 Data visualisation overview*

![][image11]

*Fig 5.2.2 Trio counts per extended family*

We dropped individuals who didn’t exist in a valid trio and removed individuals who didn’t have any images associated with them. The total number of images dropped to 13328\. The mean number of images for a father was 86, a mother was 35, and a child was 40\. In terms of top people and families by image count, Miley Cyrus has 12.1%, and similarly, the Cyrus family has 12.3%

![][image12]

*Fig 5.2.3 Average number of images for each family member*

![][image13]

*Fig 5.2.4 Most common individuals*

![][image14]

*Fig 5.2.5 Most common family names*

##

###

### 5.3 Generating Keypoints  {#5.3-generating-keypoints}

The next target was generating key points; the premise for this was to:

1. Align the images
2. To identify outliers and images that aren’t human or the people in question.

We went with the Multi-task Cascaded Convolutional Network (MTCNN) to detect key points in the image. *(Zhang et al., 2016\)* MTCNN comprises three stages. The first stage is a Proposal Network; a fully convolutional network is used to obtain candidate windows and their bounding box regression vectors. All candidates from the P-Net are fed into the Refine Network, which is a CNN that reduces the number of candidates by applying techniques such as NMS. It outputs if an input is a face or not. Finally, the Output Network outputs the five facial landmarks’ positions for eyes, nose and mouth.

![][image15]
*Fig 5.3.1 MTCNN Landmarks*

This technique also helped to identify images that were not human, too pixelated or aligned poorly as the model couldn’t detect any landmarks.

![][image16]
*Fig 5.3.2 Barack\_Obama\_20364.jpg*

![][image17]
*Fig 5.3.3 Gwyneth\_Paltrow\_0116.jpg*

![][image18]
*Fig 5.3.4 John\_Astin\_0016.jpg*

![][image19]
*Fig 5.3.5 Angelina\_Jolie\_0172.jpg*

The detector ultimately identified 97 images without landmarks, and while the majority appeared to be okay, we went and discarded all 97 images as we prioritised accuracy over recall as we had 13,000 other images. Reflecting on it now, it appears that the small resolutions of the images contributed heavily to the large ratio of false negatives. More on that later.

###

### 5.4 Calculating Face Angles  {#5.4-calculating-face-angles}

To calculate alignment in an image, three angles must be considered: yaw, pitch and roll *(Jain, 2025*). Yaw is the rotation around the vertical axis if the head turns left or right. Pitch is the side-to-side rotation about the horizontal axis. Roll is the tilt rotation if the head turns sideways.

![][image20]
*Fig 5.4.1 Pitch, roll and yaw*

In theory, we could estimate the angles using the relative positions of the key points gathered in the previous stage. Using the estimations, we could calculate if the images required alignment and manually adjust them.

We calculated the yaw by comparing the left eye to nose vs the right eye to nose. As the head rotates, one eye appears closer to the nose. Similarly, pitch was calculated by comparing the relative position of the nose to the eye-mouth line, as when your head tilts up and down, your nose appears higher or lower up your face. The roll was the angle between your eye line and the horizontal.

![][image21]
Fig 5.4.2 Face angles calculated

![][image22]
*Fig 5.4.3 Facial angles dataframe described*

After testing the data, we discovered that the values didn’t match our expectations. See the example below where we tested out a yaw of 30.0 and above, which means that the head must be turned significantly to the right; at this angle, the left eye would appear farther from the nose, and most importantly, the face would appear asymmetric. However, the second two images, Maya Soetoro and George W Bush appear to be center aligned and Obama’s image is just tilted so a high roll not yaw.

![][image23]
*Fig 5.4.4 Extracting images with yaw \>30*

![][image24]
*Fig 5.4.5 Images with yaw \>30*

###

### 5.5 Challenges with 2D Pose Estimation  {#5.5-challenges-with-2d-pose-estimation}

As shown in the images above, creating accurate yaw, pitch and roll estimations from 2D facial key points is inherently more challenging than 3D data due to the loss of depth information, projection ambiguity, individual facial differences and lighting and occlusion effects. The paper “Head Pose Estimation in the Wild using Approximated 3D Models and Deep Learning” by Ruiz et al. (2018) supports this by stating how “estimating the head pose from a single 2D image is an ill-posed problem due to the loss of depth information during camera projection, and consequently, identical 2D facial appearances can be generated from different 3D head poses.”

Therefore, we have two options: either manually create a generic 3D face model template with standard coordinates for eyes, nose and mouth landmarks. E.g, eyes at (-0.5, 0, 0\) and (0.5, 0, 0), nose at (0, \-0.3, 0.3), etc. and then to estimate and align the faces using that for comparison. Or we could use an existing 2D aligner library.

We found two repositories on github: [https://github.com/chi0tzp/FFHQFaceAlignment](https://github.com/chi0tzp/FFHQFaceAlignment) and [https://github.com/eladrich/pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel). However, when we tried implementing the alignment, their internal landmark detector was unable to detect any landmarks due to the sizing of the image. Many alignment tooling would crop the existing larger image and then align it, but in our case, the images are too small to detect landmarks for a face, let alone crop.

### 5.6 Challenges upscaling images  {#5.6-challenges-upscaling-images}

As mentioned earlier, we want our final images to be at least of resolution 256x256 to be passed into StyleGan’s encoder. Unfortunately, since we started with low-resolution images of just 120x150, we can’t just manually upscale the image.

Manual upscaling methods like bilinear, bicubic, or nearest-neighbor interpolation work by essentially "stretching" the original pixels or calculating new pixels based on mathematical formulas using the values of existing pixels *(Kim, 2025\)*. These methods follow deterministic rules that cannot invent new details, which means it can only make very basic estimations based on surrounding pixels.

This leads to two common issues: pixelation and blurriness. Nearest-neighbor interpolation simply duplicates the pixels, creating a blocky appearance, resulting in the original pixel becoming a larger square of identical pixels. Contrarily, bilinear and bicubic interpolation create smooth transitions between pixels by averaging neighboring pixel values, creating a blurry appearance.

We decided to instead go with AI Super Resolution algorithms for a variety of reasons *(Ledig et al., 2017*). They have been trained on millions of low and high resolution pairs, learning the statistical patterns of how details appear in real images. Rather than interpolating between pixels, these models synthesize realistic pixels that are optimised to look natural to the human eye.

Initially, we chose to run ESRGAN, but the corresponding results were disappointing.

![][image25]
*Fig 5.6.1 Original Image*

![][image26]
*Fig 5.6.2  ESRGAN results*

We then turned to using Real-ESRGAN, an improvement on ESRGAN that is specifically designed to handle real-world low-quality images with complex degradation (bicubic downsampling) as opposed to ESRGAN being trained on synthetic degradation. That, along with other changes like having a U-Net structured discriminator, helped it to distinguish and formulate real-world images.

![][image27]
*Fig 5.6.3 Original Image*
![][image28]
*Fig 5.6.4 Real-ESRGAN results*

The images all now have a 240x300 resolution, which makes them easier to work with when aligning.

We faced many issues at this stage due to:

1. A mismatch between the architecture between the RealESRGANer handle and the latest pre-loaded models.

![][image29]
*Fig 5.6.5 Model/RealSRGAN handle mismatch*

2. Errors due to depreciated modules in the underlying dependencies to run Real-ESRGAN like the torchvision.transforms.functional\_tensor.

![][image30]
*Fig 5.6.6 Deprecated model error*

We switched to the original model published in the Real-ESRGAN paper and manually debugged the dependency issues, updating our code to use newer modules/handlers.

### 5.7 Aligning and Blurring the Dataset {#5.7-aligning-and-blurring-the-dataset}

Once we had the upscaled images, we aligned and blurred the data using code from ([https://github.com/omertov/encoder4editing](https://github.com/omertov/encoder4editing))

![][image31]
*Fig 5.7.1 Image alignment and blurring*

Similar to our earlier implementation using MTCNN, they used a face detector called dlib and a pre-trained model file “shape\_predictor\_68\_face\_landmarks.dat”. As the name suggests, the model predicts 68 key facial points. These are spread across key facial regions:

* Jawline (points 0-16)
* Eyebrows (points 17-26)
* Nose (points 27-35)
* Eyes (points 36-47)
* Mouth (points 48-67)

It then calculates vectors between these points in a 2D plane and creates a transformation matrix to align the face based on eye and mouth positions. Finally, it crops, scales and transforms the image to standardise the face position.

Additionally, it allies Gaussian blur to create a soft transition at the edges between the edge of the image and the face, which helps the encoder to detect images better.

### 5.8 Converting into format for Latent Vector Encoding {#5.8-converting-into-format-for-latent-vector-encoding}

We now have a folder of Aligned images, however, as mentioned in the goal. We need to return to the encoder a corresponding list of mother, father and child images.

The first stage was to reassign each image to each individual.

![][image32]
*Fig 5.8.1 Reassigning images to each individual*

Then, we grouped all image paths for an individual together.

![][image33]
*Fig 5.8.2 Image path grouping*

We then repopulated the trio dataframe with the list of image paths per trio.

![][image34]
*Fig 5.8.3 Conversion to trio dataframe*

Finally, we converted the trio data into lists.

![][image35]
*Fig 5.8.4  Table to list conversion*

An interesting decision we made was how many trios we wanted to create for each nuclear family/trio. We had a few options we could choose from,   such as:

1. Calculate the number of mother, father and child images and use the minimum to create the number of trios.
2. Create the max number of unique combinations
3. Have an arbitrary set number of trios that each family should have so that we can balance contribution for each family

We went with the first option since we didn’t want to increase our number of trio images tenfold as we didn’t have the training resources for it.

6. ## **Experimentation** {#experimentation}

### 6.1 What didn’t work? {#6.1-what-didn’t-work?}

We experimented with some alternative ways of combining the parent images, learning how to edit a specific feature and optimise. The following pipeline aimed to generate child faces by combining parent images through a latent space representation while enabling targeted feature editing. The pipeline was built using StyleGAN2-ADA for face generation and a pixel2style2pixel (pSp) encoder for mapping real images into the StyleGAN latent space *(Abdal et al., 2019\)*.

The experiment proceeded as follows:

1. **Preprocessing**: Standard image processing techniques were applied to align and normalize the input images.
2. **Latent Space Encoding**: Parent images were encoded into W+ latent space using the pSp encoder.
   ![][image36]
   *Fig 6.1.1 Latent space encoding*
3. **Parent Combination**: The latent vectors of both parents were interpolated using spherical linear interpolation (SLERP) to produce a combined latent representation.
   ![][image37]
   *Fig 6.1.2 Parent combination*
4. **Feature Extraction**: Facial attributes were extracted using the *facer* library to quantify specific traits.
   ![][image38]![][image39]![][image40]
   *Fig 6.1.3  Kareena Kapoor Feature Extraction using Facer*
5. **Feature Model Training**: Regression and classification models were trained to map facial features to directional changes in the latent space.
6. **Feature-Based Editing**: Functions were developed to apply modifications to latent vectors based on predefined strengths of specific features.
   ![][image41]
   *Fig 6.1.4 Facial feature manipulation from latent-features mapping*
7. **Optimization Loop**: An iterative process was used to determine the best strength values for feature modifications by minimizing the difference between generated and real child attributes using a loss function.

###

### 6.2 Metrics {#6.2-metrics}

Given the mixed nature of facial attributes (both continuous and categorical), different evaluation metrics were chosen:

* **Regression tasks** (e.g., age, face roundness) used **R² score** to assess the variance explained by the model.
* **Classification tasks** (e.g., gender, presence of glasses) used **accuracy** as the primary metric.
* The **loss function** for optimization was **mean squared error (MSE)** between the generated child attributes and real child attributes.

### 6.3 Results {#6.3-results}

Several visualizations were created to analyze the performance of the generated child faces:

* **Original vs. generated parents**: A comparison of real parent images with their latent-space reconstructions.
* **Feature manipulation demonstrations**: Side-by-side images showing how modifying individual features (e.g., smiling, age progression) affected the generated faces.
* **Before/after comparisons of parent combinations and their generated child images**.

#### **Observations:** {#observations:}

* **Small Face Problem**: Generated child faces often appeared unnaturally small, likely due to latent space interpolation effects.
  ![][image42]
  *Fig 6.3.1 Combined parent faces were disproportionately sized compared to their head*

* **Loss of Detail**: Averaging parent features resulted in a blurring of defining characteristics, reducing the distinctiveness of generated faces.
  ![][image43]![][image44]![][image45]
  *Fig 6.3.2  Joseph P Kennedy & Rose Elizabeth Fitzgerald latents combined*
  ![][image46]![][image47]![][image48]
  *Fig 6.3.3 Alan Hale Sr & Gretchen Hartman converted to latent vector and averaged*
* **Feature Editing Issues**: Modifying facial features post-hoc in latent space did not yield effective transformations, possibly due to the limited sample size.
* **Encoder Artifacts**: Manually encoding real images produced distorted faces, suggesting that a more refined encoding approach might be necessary.
  ![][image49]![][image50]
  *Fig 6.3.4 Distorted images of Barack OBama, after encoding to a latent vector and converting back to an image.*
* **Feature Strength Optimization Issues**:
  * The model might be learning non-informative features, e.g., *wearing a necklace*, instead of relevant facial characteristics.
  * Limited training data could be restricting the effectiveness of the feature strength adjustments.
  * Overly strong edits caused child faces to become too uniform, lacking natural variation.

![][image51]![][image52]

*Fig 6.3.5 Two generated children, derived from completely distinct parents*

##

##

## 6.4 Conclusion

This experimental attempt highlighted key challenges in latent space-based facial feature manipulation. While the approach showed promise in interpolating parent features, it suffered from issues related to face size, detail retention, and effective feature modification. Future improvements may involve:

* Using a **better encoder** from prior research to avoid artifacts.
* Increasing the dataset size for **better feature model training**.
* Refining the **feature strength optimization process** to avoid excessive uniformity.

Overall, this experiment provided valuable insights into the complexities of genetic-inspired facial synthesis and the challenges of latent space manipulation.

We found that the inaccuracies associated with converting to latent vector and combining parent faces could be avoided by using methods found in research papers rather than attempting to do this manually.

7. ## **ML Model / Network Structure** {#ml-model-/-network-structure}

![][image53]
*Fig 7.1 Pipeline structure*

### 7.1 StyleGAN Architecture and Implementation {#7.1-stylegan-architecture-and-implementation}

For our kinship face generation project, we selected StyleGAN2 as our foundational model due to its superior quality in facial image generation and well-structured latent space. StyleGAN2 offers significant improvements over the original StyleGAN, including path length regularization and redesigned generator normalization *(Karras et al., 2020\)*, which result in higher fidelity face generation with fewer artifacts.

#### 7.1.1 Model Selection {#7.1.1-model-selection}

We explored several GAN architectures for face generation:

1. **StyleGAN1**: While pioneering the style-based approach, it often produced "blob" artifacts in generated faces.
2. **StyleGAN2**: Offers improved image quality, better disentanglement of facial features, and eliminates many artifacts.
3. **StyleGAN3**: Provides rotational equivariance, but StyleGAN2 was deemed more suitable for our specific task of kinship face synthesis due to its well-established latent space manipulation techniques.

#### 7.1.2 Encoder Selection {#7.1.2-encoder-selection}

After evaluating several encoding approaches, we selected the Encoder4Editing (e4e) encoder for projecting real face images into StyleGAN2's latent space *(Richardson et al., 2021\)*. Considerations were given to:

1. **pSp (pixel2style2pixel)**: Provides high-quality reconstructions but often results in less editable latent codes. (Richardson et al, CVPR 2021\)
2. **HyperStyle**: Offers state-of-the-art reconstruction quality but requires additional computational resources. (Alaluf et al, CVPR 2022\)
3. **Stylegan2encoder** (ResNet-based model): Utilizes a ResNet architecture to encode images into StyleGAN2's latent space, enabling effective reconstructions but may require fine-tuning for optimal performance. ​ (Luxemburg 2020\)

The e4e encoder maps real parent images to the W+ latent space while maintaining editability, which is essential for our subsequent kinship feature blending operations. In order to leverage the e4e encoder, we developed a wrapper library around the core features of the e4e encoder repo.

### 7.2 Proposed Kinship Synthesis Model {#7.2-proposed-kinship-synthesis-model}

Our model consists of several components that work together to generate realistic child faces from parent images:

#### 7.2.1 Parent Image Embedding {#7.2.1-parent-image-embedding}

We use the e4e encoder to project parent images into StyleGAN2's W+ latent space:

| parent\_latent \= e4e\_encoder(parent\_image) |
| :---- |

This step transforms each parent's facial image into a set of latent vectors (W+) that StyleGAN2 can manipulate and regenerate.

#### 7.2.2 Latent Space Blending {#7.2.2-latent-space-blending}

For combining parental features, we implement a weighted blending of latent codes in W+ space *(Karras et al., 2020\)*:

In our implementation, we utilize several sophisticated approaches for combining parental latent codes in the W+ space of StyleGAN2. These methods allow us to create child faces that inherit specific facial features from each parent in a manner that mimics genetic inheritance patterns.

##### 7.2.2.1 Simple Weighted Average {#7.2.2.1-simple-weighted-average}

The foundation of our latent space blending is a weighted average approach that combines the mother's and father's latent codes:

| def combine\_latents(mother\_latent, father\_latent, weights=None):    if weights is None:        *\# Default 50-50 blend*        weights \= \[0.5, 0.5\]        combined\_latent \= weights\[0\] \* mother\_latent \+ weights\[1\] \* father\_latent    return combined\_latent |
| :---- |

This basic approach serves as our baseline, creating a uniform blend where each parent contributes equally (or at specified proportions) to all facial features. While simple, this method provides surprisingly realistic results in many cases, especially when parents share similar facial structures.

##### 7.2.2.2 Per-Dimension Weighting {#7.2.2.2-per-dimension-weighting}

To achieve more precise control over feature inheritance, we implement dimension-specific weighting that targets individual components of the latent code:

| \# Application of per-dimension weightsfor dim, weight in dimension\_weights.items():    \# Apply weight to specific dimensions across all layers    weight\_1\[:, dim\] \= weight    weight\_2\[:, dim\] \= 1.0 \- weight |
| :---- |

This approach allows us to specify different inheritance patterns for specific facial attributes. For example, given enough testing and experimentation to identify boundaries, we can set the model to inherit eye shape predominantly from the mother while taking nose structure from the father. Our experiments revealed that certain latent dimensions correspond to specific facial features, allowing targeted manipulation.

##### 7.2.2.3 Layer-Wise Weighting {#7.2.2.3-layer-wise-weighting}

StyleGAN's hierarchical generation process means different layers control different levels of detail. Our layer-wise weighting exploits this property:

| *\# Application of layer-specific weights*for layer, weight in layer\_weights.items():    *\# Apply weight to all dimensions in the specified layer*    weight\_1\[layer, :\] \= weight    weight\_2\[layer, :\] \= 1.0 \- weight |
| :---- |

This technique allows control over whether a child inherits coarse features (controlled by early layers) or fine details (controlled by later layers) from each parent. For instance, we can blend face shape from the father with skin texture and fine details from the mother.

##### 7.2.2.4 Binary Mask Blending {#7.2.2.4-binary-mask-blending}

In addition, to attempt to better mimic the discrete nature of genetic inheritance, we implemented a binary mask approach:

| def combine\_latent\_dimensions(latent\_1, latent\_2, blend\_mask=None):    *\# Create random binary mask if not provided*    if blend\_mask is None:        blend\_mask \= torch.randint(0, 2, latent\_shape).float()        *\# Combine latents using the mask*    combined\_latent \= latent\_1 \* blend\_mask \+ latent\_2 \* (1 \- blend\_mask)        return combined\_latent |
| :---- |

This method selects each feature discretely from either the mother or father, rather than blending them. The binary mask can be random (mimicking the stochastic nature of genetic inheritance) or designed to target specific facial regions.

#### 7.2.3 Feature Refinement with InterfaceGAN {#7.2.3-feature-refinement-with-interfacegan}

To fine-tune specific attributes in the generated child faces, we leverage pretrained InterfaceGAN boundaries for controlled attribute manipulation *(Shen et al., 2020\)*:

| *\# Applying attribute edits using InterfaceGAN*def apply\_attribute\_edit(latent, direction\_name, strength):    direction \= load\_direction(direction\_name)  *\# Load pretrained direction*    edited\_latent \= latent \+ direction \* strength    return edited\_latent |
| :---- |

These pretrained boundaries allow us to adjust attributes like age, gender expression, and other facial characteristics to achieve more realistic resemblance.

### 7.3 Neural Optimization Framework for Kinship Face Generation {#7.3-neural-optimization-framework-for-kinship-face-generation}

#### 7.3.1 Overview {#7.3.1-overview}

The neural optimization framework represents a key innovation in our approach to kinship face synthesis. Rather than relying on fixed or manually-tuned blending weights, we developed a trainable neural network that learns to predict optimal blending parameters based on the specific characteristics of parent pairs *(Goodfellow et al., 2014\)*. This approach allows the system to develop a deeper understanding of genetic inheritance patterns as they relate to facial features.

#### 7.3.2 Architecture and Implementation {#7.3.2-architecture-and-implementation}

The neural optimization framework consists of two primary components:

##### 7.3.2.1 Adaptive Dimension Weight Model {#7.3.2.1-adaptive-dimension-weight-model}

| class AdaptiveDimensionWeightModel(nn.Module):    def \_\_init\_\_(self, latent\_shape):        super(AdaptiveDimensionWeightModel, self).\_\_init\_\_()               self.latent\_shape \= latent\_shape        self.num\_layers \= latent\_shape\[0\]        self.latent\_dim \= latent\_shape\[1\]                *\# Total size of flattened latent*        latent\_size \= self.num\_layers \* self.latent\_dim        input\_dim \= 2 \* latent\_size  *\# Both parents' full latent codes*                *\# Build encoder for feature extraction*        self.encoder \= nn.Sequential(            nn.Linear(input\_dim, 2048),            nn.LeakyReLU(0.2),            nn.BatchNorm1d(2048),            nn.Dropout(0.3),                        nn.Linear(2048, 1024),            nn.LeakyReLU(0.2),            nn.BatchNorm1d(1024),            nn.Dropout(0.3),                        nn.Linear(1024, 512),            nn.LeakyReLU(0.2),            nn.BatchNorm1d(512)        )                *\# Attention module to highlight important features*        self.attention \= nn.Sequential(            nn.Linear(input\_dim, 256),            nn.ReLU(),            nn.Linear(256, input\_dim),            nn.Sigmoid()        )                *\# Weight decoder for generating latent weights*        self.weight\_decoder \= nn.Sequential(            nn.Linear(512, 1024),            nn.LeakyReLU(0.2),            nn.BatchNorm1d(1024),                       nn.Linear(1024, 2048),            nn.LeakyReLU(0.2),            nn.BatchNorm1d(2048),                        nn.Linear(2048, latent\_size),            nn.Sigmoid()  *\# Output weights between 0 and 1*        ) |
| :---- |

###### *Model Explanation* {#model-explanation}

the network architecture begins with three primary components:

- **Encoder**: Processes the concatenated parent latent codes through several linear layers with LeakyReLU activations, batch normalization, and dropout. It compresses the input information into a 512-dimensional feature space.
- **Attention module**: Applies a self-attention mechanism to focus on the most relevant features of the parent latents. The sigmoid activation ensures attention weights are between 0 and 1\.  Vaswani et al., "Attention Is All You Need," NeurIPS 2017 (foundational attention mechanism paper, not provided in search results but should be cited accordingly).
- **Weight decoder**: Expands the compressed representation back to the original latent size, with the final sigmoid ensuring all weight values are between 0 and 1\.

The model effectively learns which features to take from each parent for every dimension in the latent space, allowing for fine-grained control over inheritance patterns.

##### 7.3.2.2 Differentiable Training Pipeline {#7.3.2.2-differentiable-training-pipeline}

To enable end-to-end training, we had to implement a fully differentiable pipeline that includes:

| class MockProcessor:    """    A mock processor to use during training that maintains gradient flow.    Instead of generating actual images, it simulates the process in a differentiable way.    """    def \_\_init\_\_(self, face\_encoder):        self.face\_encoder \= face\_encoder        self.register\_buffers()            def generate\_embeddings\_from\_latent(self, latent):        """        Generate face embeddings directly from latent code in a differentiable way.        """        *\# Create a stable differentiable transformation from latent to embedding space*        batch\_size \= latent.size(0) if latent.dim() \> 2 else 1        if latent.dim() \== 2:            latent \= latent.unsqueeze(0)                *\# Get device dynamically*        device \= latent.device        if not hasattr(self, 'projection\_matrix') or self.projection\_matrix.device \!= device:            self.register\_buffers()                    *\# Flatten the latent code*        h \= latent.view(batch\_size, \-1)  *\# \[batch\_size, 18\*512\]*                *\# Apply stable linear projection with a fixed matrix*        if self.projection\_matrix.device \!= h.device:            self.projection\_matrix \= self.projection\_matrix.to(h.device)                    h \= F.linear(h, self.projection\_matrix)  *\# \[batch\_size, 512\]*                *\# Apply non-linearity and normalization*        h \= F.relu(h)        h \= F.normalize(h, p=2, dim=1)  *\# Normalize to unit length like real embeddings*                return h |
| :---- |

The mock processor is required as a differentiable bridge in the model. As StyleGAN's image generation process is inherently non-differentiable, this would normally prevent gradient-based training of the latent weight generator. The mock processor solves this problem by creating a simplified, differentiable approximation that maps directly from StyleGAN latent codes to face embedding space, bypassing the actual image generation and face recognition steps. Rather than attempting to replicate the complex internal processing of multiple neural networks (StyleGAN's generator followed by a ResNet-based face recognition model), it implements a static randomly generated matrix that allows gradients to flow from the target embeddings back to the weight generator during training.

This approach works because the system doesn't need to understand exactly how latents become images and then embeddings \- it only needs to learn the relationship between latent code combinations and the resulting face similarity measurements. By training with real face embeddings as targets, the weight generator learns to produce latent combinations that, when passed through the real StyleGAN generator (during inference), will create images with facial features similar to actual children. The mock processor thus acts as a computational shortcut during training.

### 7.4 Training Methodology {#7.4-training-methodology}

#### 7.4.1 Loss Function {#7.4.1-loss-function}

We implemented a face similarity loss that measures the perceptual similarity between generated child embeddings and real child face embeddings:

| def \_compute\_face\_similarity\_loss(self, generated\_embedding, target\_embedding):    """    Compute perceptual loss between generated embedding and target embedding.    """    *\# Always ensure both embeddings are normalized for consistent similarity metrics*    generated\_embedding \= F.normalize(generated\_embedding, p=2, dim=1)        *\# Ensure target embedding is also normalized*    if target\_embedding.requires\_grad:        target\_embedding \= F.normalize(target\_embedding, p=2, dim=1)    else:        with torch.no\_grad():            target\_embedding \= F.normalize(target\_embedding, p=2, dim=1)        *\# Compute cosine similarity (higher is better)*    cos\_sim \= F.cosine\_similarity(generated\_embedding, target\_embedding)        *\# Compute L2 distance (lower is better)*    l2\_dist \= torch.sum((generated\_embedding \- target\_embedding) \*\* 2, dim=1)        *\# Weighted combination*    loss \= (1.0 \- cos\_sim) \+ 0.05 \* l2\_dist        return loss |
| :---- |

This loss function operates as follows:

- **Input**: Takes face embeddings from the generated child face and the real child face
- **Normalization**: Ensures both embeddings are normalized to unit length (for consistent similarity metrics)
- **Cosine Similarity**: Calculates how aligned the two embeddings are in the feature space (value between \-1 and 1, where 1 means identical direction)
- **L2 Distance**: Calculates the squared Euclidean distance between the embeddings
- **Combined Loss**: Creates a loss value that:
  - Decreases as cosine similarity increases (using `1.0 - cos_sim`)
  - Increases with L2 distance, but with a lower weight (0.05) to avoid dominating the loss
  - This dual approach provides more stable gradients than either metric alone

This loss effectively encourages the model to generate child faces that have similar facial features to the real child, rather than just similar pixel values.

#### 7.4.2 Training Process {#7.4.2-training-process}

The training procedure involves:

| def train\_epoch(self, dataloader, child\_embeddings):    """Train for one epoch."""    *\# Ensure face encoder and mock processor are initialized*    self.\_initialize\_face\_encoder()        self.model.train()    total\_loss \= 0    valid\_batches \= 0        for batch in tqdm(dataloader, desc="Training"):        *\# Get data and move to device*        father\_latent \= batch\['father\_latent'\].to(self.device)        mother\_latent \= batch\['mother\_latent'\].to(self.device)        original\_indices \= batch\['original\_idx'\].tolist()                *\# Zero gradients*        self.optimizer.zero\_grad()                *\# Forward pass \- get weights*        weights \= self.model(father\_latent, mother\_latent)                *\# Combine latents using weights*        predicted\_child\_latent \= self.\_combine\_latents\_with\_weights(            father\_latent, mother\_latent, weights        )                *\# Calculate batch loss*        batch\_loss \= 0.0        valid\_samples \= 0                *\# Process each sample in the batch*        for i in range(father\_latent.size(0)):            original\_idx \= original\_indices\[i\]            target\_embedding \= child\_embeddings\[original\_idx\]                      if target\_embedding is not None:                *\# Generate embeddings directly (differentiable pathway)*                sample\_latent \= predicted\_child\_latent\[i\].unsqueeze(0)                generated\_embedding \= self.mock\_processor.generate\_embeddings\_from\_latent(sample\_latent)                                *\# Compute perceptual loss*                sample\_loss \= self.\_compute\_face\_similarity\_loss(generated\_embedding, target\_embedding)                batch\_loss \+= sample\_loss                valid\_samples \+= 1                *\# Average loss over valid samples in batch*        if valid\_samples \> 0:            batch\_loss \= batch\_loss / valid\_samples                        *\# Backward pass*            batch\_loss.backward()                        *\# Apply gradient clipping*            torch.nn.utils.clip\_grad\_norm\_(self.model.parameters(), self.clip\_value)                        *\# Optimize*            self.optimizer.step()                      *\# Update tracking metrics*            total\_loss \+= batch\_loss.item()            valid\_batches \+= 1 |
| :---- |

##### Code Explanation {#code-explanation}

The `train_epoch` method implements one training epoch:

1. **Setup**: Initializes the face encoder and sets the model to training mode
2. **Batch Processing**: For each batch of parent pairs:
   - **Data Preparation**: Loads father and mother latent codes and their original indices
   - **Forward Pass**:
     - Runs the latent pairs through the model to predict blending weights
     - Combines the parent latents using these weights via weighted sum: `father_latent * weights + mother_latent * (1 - weights)`
   - **Individual Sample Processing**: For each pair in the batch:
     - Retrieves the target child embedding using the original index
     - Uses the mock processor to generate a face embedding from the predicted child latent
     - Computes the face similarity loss between generated and real child embeddings
   - **Optimization**:
     - Averages the loss across valid samples
     - Performs backpropagation
     - Applies gradient clipping to prevent extreme updates
     - Updates the model parameters using the optimizer
     - Tracks the total loss for reporting

This training process effectively teaches the model to predict weights that will produce a latent code that generates a face similar to the real child when passed through the face embedding network.

#### 7.4.3 Latent Combination Mechanism {#7.4.3-latent-combination-mechanism}

The core of child face generation is the latent combination function:

| def \_combine\_latents\_with\_weights(self, father\_latent, mother\_latent, weights):    *\# weights tensor represents the weight for father's contribution*    *\# (1 \- weights) is implicitly the weight for mother's contribution*    return father\_latent \* weights \+ mother\_latent \* (1 \- weights) |
| :---- |

##### Code Explanation {#code-explanation-1}

This function:

- Takes the father and mother latent codes along with the predicted weights
- Performs element-wise multiplication of each parent's latent with its corresponding weight
- The weights tensor represents the father's contribution proportion
- `(1 - weights)` automatically becomes the mother's contribution proportion
- Adds the weighted latents together to produce a combined child latent
- Each dimension of the latent space is weighted independently, allowing for fine-grained control over feature inheritance

This weighted combination preserves the latent space structure while creating a valid new point that represents the child face.

#### 7.4.4. Overfitting and attempts to combat it {#7.4.4.-overfitting-and-attempts-to-combat-it}

![][image54]
*Fig 7.4.4 Overfitting*
Our neural network for predicting optimal parent-child blending weights showed clear signs of overfitting around epoch 40, with validation loss remaining stagnant while training loss decreased. Beyond the techniques already mentioned (dropout, batch normalization, etc.), we implemented several additional strategies to combat this issue:

Beyond standard dropout and weight decay, we utilized:
**Gradient Clipping**: We implemented conservative gradient clipping to prevent extreme weight updates that could lead to memorization:

**Adaptive Weight Decay**: The AdamW optimizer was configured with a weight decay value to impose stronger regularization as the model grew more complex.

**Progressive Dropout Adjustment**: We experimented with increasing dropout rates throughout training, starting with lower values (0.2) and progressively increasing them up to 0.5 in later epochs when overfitting became more pronounced.

**Latent Space Noise Injection**: During training, we experimented with adding small amounts of random noise to parent latent codes, creating subtle variations that helped the model generalize better:

| if self.training\_noise \> 0:    noise\_factor \= self.training\_noise \* (1.0 \- epoch / total\_epochs)  *\# Decay over time*    father\_latent \= father\_latent \+ torch.randn\_like(father\_latent) \* noise\_factor    mother\_latent \= mother\_latent \+ torch.randn\_like(mother\_latent) \* noise\_factor |
| :---- |

To ensure we selected the optimal model checkpoint we maintained an exponential moving average of model weights during training, which typically generalizes better than the final model weights:

| *\# Update moving average model*if moving\_avg\_model is None:    moving\_avg\_model \= {k: v.cpu().clone() for k, v in self.model.state\_dict().items()}else:    for k, v in self.model.state\_dict().items():        moving\_avg\_model\[k\] \= moving\_avg\_model\[k\] \* 0.9 \+ v.cpu() \* 0.1 |
| :---- |

However, on repeat training attempts, none of these alterations were able to significantly reduce the impact of overfitting. This would indicate to us that additional attention may be needed to be taken into the ‘data side’ of overfitting. This may involve gathering additional datasets or performing alternative data augmentation to diversify our dataset.

8. ## Results and Evaluation {#results-and-evaluation}

### 8.1 Data {#8.1-data}

![][image55]

*Fig 8.1.1 Average weight per StyleGAN layer*

![][image56]

*Fig 8.1.2 Model-learned parent weight heatmap*

**Most consistent (layer, dimension) pairs across samples:**

| Layer | Dimension | Variance | Avg Weight |
| :---- | :---- | :---- | :---- |
| 10 | 186 | 0.000134 | 0.0100 |
| 12 | 405 | 0.000175 | 0.0068 |
| 14 | 189 | 0.000221 | 0.0092 |
| 15 | 364 | 0.000225 | 0.0099 |
| 6 | 141 | 0.000534 | 0.9740 |
| 16 | 228 | 0.000599 | 0.0259 |
| 8 | 490 | 0.000755 | 0.9571 |
| 7 | 15 | 0.000764 | 0.0320 |
| 14 | 21 | 0.000775 | 0.0264 |
| 9 | 15 | 0.000776 | 0.0432 |
| 11 | 377 | 0.000851 | 0.9755 |
| 16 | 341 | 0.001154 | 0.0207 |
| 14 | 341 | 0.001274 | 0.0129 |
| 6 | 171 | 0.001447 | 0.0334 |
| 4 | 244 | 0.001522 | 0.9820 |
| 6 | 185 | 0.001593 | 0.0691 |
| 15 | 199 | 0.001705 | 0.0456 |
| 15 | 189 | 0.001875 | 0.9729 |
| 12 | 67 | 0.002072 | 0.0381 |
| 17 | 20 | 0.002118 | 0.9631 |

**Overall parent bias:** 0.4995 (mother dominance)
**Average weight deviation from 0.5:** 0.3488

Our analysis of the model's learned blending weights reveals two significant patterns that mirror natural genetic inheritance mechanisms

**Global Balance with Feature-Specific Inheritance**: On the overall, our parent bias of 0.4995 demonstrates that the model learned to take almost exactly an equal amount from both of the parents on average. But, the large average deviation (0.3488) from the midpoint seems to indicate that for specific facial features, the model strongly prefers one parent over the other, akin to how genetic dominance operates in real biological inheritance.

**Feature-Specific Consistency**: The table above shows specific layer-dimension pairs with remarkably low variance across different family samples, suggesting that certain facial features consistently come from either the mother or father across different families. For example, dimension 141 in layer 6 shows a strong father bias (0.9740) with very low variance (0.000534), potentially corresponding to a specific facial feature that tends to be paternally inherited.

From a high-level, the neural optimization approach theoretically offers several advantages over fixed or heuristic blending methods:

1. **Context-awareness**: The model learned to adjust blending weights based on specific parent features, rather than applying the same weights for all parent pairs.
2. **Feature correlation understanding**: The network implicitly discovered which facial features tend to be inherited together.
3. **Genetic inheritance patterns**: The model learned patterns resembling dominant/recessive inheritance, where some facial features from one parent are strongly expressed in the child.

In addition, during qualitative assessments, child faces generated with the neural optimizer showed:

* Better preservation of distinctive family traits
* More natural blending of features that tend to be inherited together
* Reduced facial structure artifacts compared to simple averaging methods

### **8.2 Challenges in Accuracy Assessment** {#8.2-challenges-in-accuracy-assessment}

There is a fundamental challenge in evaluating our model. It arises from the inherent randomness and complexity of genetic inheritance itself. Unlike classification tasks with clear ground truths, there is no single "correct" face that should result from two parents. Even when given a large sample set of one family's children, assessing exactly what the ‘truth’ should be is impossible.

 In real genetics:

1. Genetic recombination shuffles genes between chromosomes
2. Complex interactions between dominant and recessive alleles determine trait expression
3. Many facial features are polygenic (controlled by multiple genes)

This genetic variability explains why biological siblings can look completely different *despite* sharing the same parents *(Visscher et al., 2008\)*. Given this inherent randomness, traditional accuracy metrics become inadequate for evaluating our model. A more appropriate assessment might involve qualitative analysis of perceived family resemblance and biological plausibility of the generated faces or an analysis of known dominant/recessive genes relationships (e.g. brown/blue eyes).

Our model's tendency to strongly favor one parent for specific features rather than creating bland averages actually mirrors real genetic inheritance more closely than a simple 50/50 blend would. This suggests that the neural network has in some capacity learned aspects of the genetic inheritance patterns within the training data, in spite of the fact it had not been explicitly programmed with biological principles.

### 8.3 Technical Challenges and Memory Optimization {#8.3-technical-challenges-and-memory-optimization}

One of the most significant challenges in implementing our kinship face generation system was managing GPU memory constraints. StyleGAN2, while powerful, is notably resource-intensive due to its complex architecture and the high-dimensional latent spaces it operates in. When combined with the e4e encoder, memory requirements became a critical bottleneck that threatened the viability of our entire pipeline.

#### 8.3.1 Memory Requirements and Initial Failures {#8.3.1-memory-requirements-and-initial-failures}

Our initial implementation attempts faced immediate failures with standard "CUDA out of memory" errors.  The relatively high GPU memory requirements of both stylegan2 often exceeded available resources.

#### 8.3.2 Memory Reduction Techniques {#8.3.2-memory-reduction-techniques}

To address these challenges, we attempted to implement a set of memory optimization techniques:

##### Mixed Precision Training/Inference {#mixed-precision-training/inference}

We initially added support for PyTorch's automatic mixed precision (FP16), which typically reduces memory usage by up to 50% by using half-precision floating point numbers where appropriate. This was implemented with a configurable option:

| self.inference.use\_mixed\_precision \= enable\_mixed\_precision |
| :---- |

However, we discovered that mixed precision caused critical issues with our particular model. StyleGAN2 and e4e require consistent tensor data types throughout their execution pipeline. The model was originally trained with full precision weights, and when some operations were performed in FP16 while others remained in FP32, we encountered dtype mismatches between tensors.

After debugging, we determined that the custom CUDA kernels in StyleGAN2 were not compatible with mixed precision without substantial modification to the codebase. We therefore reverted to full precision processing with explicit memory management approaches instead.

##### Memory-Efficient Model Loading {#memory-efficient-model-loading}

We implemented a "memory-efficient mode" that keeps most of the model on CPU and only transfers specific components to GPU when needed *(Micikevicius et al., 2018\)*:

| if self.memory\_efficient and device \== "cuda":    \# Move required components to GPU for the forward pass    self.net.to(device)    was\_on\_cpu \= Trueelse:    was\_on\_cpu \= False\# Process inputimages, latents \= self.net(input\_tensor, randomize\_noise=False, return\_latents=True)\# Move model back to CPU if it was movedif was\_on\_cpu:    self.net.to(*'cpu')*    \# Ensure results stay on device    images \= images.to(device)    latents \= latents.to(device) |
| :---- |

This significantly reduced the persistent GPU memory footprint, allowing us to process higher resolution images and more complex facial features.

##### Explicit Memory Management {#explicit-memory-management}

We implemented systematic memory cleanup throughout our pipeline:

Added explicit calls to `torch.cuda.empty_cache()` after GPU-intensive operations
Integrated Python's garbage collection to free unused memory:

| import gcgc.collect()torch.cuda.empty\_cache() |
| :---- |

And set PyTorch environment variable to avoid memory fragmentation:

| os.environ\['PYTORCH\_CUDA\_ALLOC\_CONF'\] \= 'expandable\_segments:True' |
| :---- |

This proactive memory management approach prevented cumulative memory buildup during sequential operations.

#### 8.3.3 Results of Memory Optimization {#8.3.3-results-of-memory-optimization}

These optimization techniques reduced our memory usage by enough to allow for e4e to be run on a consumer grade GPU. However, our training models and the use of InterfaceGan still require a T4 gpu at a minimum.

9. # Experiments & Evaluation {#experiments-&-evaluation}

### **Fairness & Bias Considerations** {#fairness-&-bias-considerations}

We’ve leveraged StyleGAN2's latent space for generating child faces from parent images. While powerful, this approach inherits biases present in the training data *(Karras et al., 2020\)*. As our implementation relies primarily on the version of stylegan2 which was trained on the FFHQ dataset (Flickr-Faces-HQ), we maintain all of the biases present in this data.
FFHQ, while diverse, still contains imbalances in terms of, age distribution (skewed toward adults), and racial representation (predominantly Caucasian) \- Maluleke et al. (2022)
In addition and as mentioned previously the Family101 dataset used for training the blending of weights also has demographic imbalances.

These hinder the effectiveness and exemplifies the biases innate in this process. For example, experimentation with the e4e encoder on people of mixed descent resulted in poor results.


10. # HCI \- A web app {#hci---a-web-app}

![][image57]

*Fig 10.1 \- Flask web app*

The web interface component of the Genetic Face Generator is implemented as a Flask-based web application, providing an intuitive and accessible front-end for the face blending technology. It offers a user-friendly interface where users can upload two parent face images and generate a child face using genetic blending algorithms. The interface features a clean, responsive design with separate upload sections for each parent image, option toggles for uniform weighting and age adjustment, and a results display showing all three images (both parents and the generated child) side by side.

The implementation uses Flask for server-side processing. The interface provides immediate visual feedback with image previews and includes error handling with user-friendly notifications. Users can customize the generation process by selecting options like uniform 50/50 weighting (versus model-based weighting) and age adjustment to make the child appear younger. The generated child image can be downloaded directly from the interface, allowing users to save their results.

11. # Discussion and Conclusion {#discussion-and-conclusion}

![][image58]
*Fig 11.1 Parent combination \- weighted and non weighted generated child*
![][image59]![][image60]
*Fig 11.2 Example child generation with non dataset images*

Our project developed a system for generating realistic child faces from parent images using a combination of StyleGAN2's latent space manipulation and a neural optimizer for intelligent feature blending.

## 11.1 Key Findings {#11.1-key-findings}

Our neural optimization approach for latent space blending produced several noteworthy findings:

1. **Feature-Specific Inheritance:** The model learned to strongly favor one parent for specific facial features rather than creating bland averages. This mirrors real genetic inheritance patterns where certain traits exhibit dominance. For example, our analysis revealed specific latent dimensions (such as dimension 141 in layer 6\) that consistently showed strong parental bias with low variance across different families.

2. **Global Balance:** Despite the feature-specific inheritance patterns, the overall parent bias in our model was remarkably balanced (0.4995), indicating no systematic preference for either mother or father across the complete dataset.

3. **Layer-Specific Patterns:** Different StyleGAN layers (controlling different levels of detail from coarse to fine) showed distinct inheritance patterns, suggesting that certain facial attributes might follow different inheritance rules than others.

## 11.2 Technical Challenges {#11.2-technical-challenges}

We encountered several significant challenges throughout the project, one of which being memory management**.** Both StyleGAN2 and the e4e encoder are both resource-intensive models that required memory optimization to become practically functional on consumer machines. Our initial implementation attempts faced immediate failures with "CUDA out of memory" errors. We successfully addressed some of this through techniques like memory-efficient model loading, explicit memory management, and modular processing but were unable to curb this issue entirely.

In addition, overfitting became an issue**.** Despite implementing various regularization techniques (dropout, batch normalization, weight decay, gradient clipping), our neural optimizer showed signs of overfitting around epoch 40\. This suggests the need for larger and more diverse training datasets.

We demonstrated that latent space integrity is challenging to maintain. Our initial approaches to latent space interpolation often resulted in unrealistic face shapes or features. Developing appropriate blending mechanisms that maintained the integrity of the StyleGAN latent space was critical to generating realistic results.

Finally, Preprocessing Complexity. Aligning and normalizing the facial images in our dataset proved challenging due to the varying quality and composition of the source images found in our dataset. Our initial attempts at calculating face angles from 2D landmarks had limitations that required us to use more sophisticated alignment methods.

## 11.3 Future Work {#11.3-future-work}

Based on our findings and limitations, several promising directions for future work emerge:

**Feature Interpretability:** Further investigation into mapping specific latent dimensions to interpretable facial features would enhance our understanding of the model's learned inheritance patterns. This may involve boundary training at a larger scale, or simple experimentation.

**Expanded Datasets:** Incorporating larger and more diverse family datasets would improve both the performance and fairness of the model across different demographics.

**Enhanced genetic-based loss function:** Develop methodologies to assess not just facial similarly, but genetic similarity for the development of an enhanced loss function.

## 11.4 Conclusion {#11.4-conclusion}

Our project demonstrates the potential of neural-optimized latent space blending for kinship face generation. By leveraging StyleGAN2's capabilities as both a face generator and manipulator, with a trainable weight prediction network, we created a system that can produce realistic child faces while mimicking aspects of genetic inheritance.

Our implementation balances overall parental contribution all the while allowing for feature-specific inheritance patterns, similar to genetic dominance in real biological systems. Despite challenges with memory management, encoder limitations, and dataset biases, our approach produces compelling results.

Future work should focus on improving interpretability, expanding dataset diversity, and developing more loss function techniques.

12. # References {#references}

Abdal, R., Qin, Y. and Wonka, P. (2019) Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space? \[online\] arXiv.org. Available at: https://arxiv.org/abs/1904.03189 \[Accessed 16 Mar. 2025\].

Chiu, P.-Y., Wu, D.-J., Chu, P.-H., Hsu, C.-H., Chiu, H.-C., Wang, C.-Y. and Chen, J.-C. (2024) StyleDiT: A Unified Framework for Diverse Child and Partner Faces Synthesis with Style Latent Diffusion Transformer. \[online\] arXiv.org. Available at: https://arxiv.org/abs/2412.10785 \[Accessed 10 Mar. 2025\].

Emara, M.M., Farouk, M. and Fakhr, M.W. (2024) Parent GAN: Image Generation Model for Creating Parent’s Images Using Children’s Images. Multimedia Tools and Applications. Available at: https://doi.org/10.1007/s11042-024-20186-y \[Accessed 4 Mar. 2025\].

Fang, R., Gallagher, A.C., Chen, T. and Loui, A. (2013) Kinship classification by modeling facial feature heredity. 2013 IEEE International Conference on Image Processing (ICIP), pp.2983–2987. Available at: https://ieeexplore.ieee.org/document/6738642 \[Accessed 1 Mar. 2025\].

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. and Bengio, Y. (2014) Generative adversarial nets. Advances in Neural Information Processing Systems (NeurIPS), 27\. Available at: https://proceedings.neurips.cc/paper/2014/hash/5ca3e9b122f61f8f06494c97b1afccf3-Abstract.html \[Accessed 22 Jan. 2025\].

Gradilla, R. (2020) Multi-task Cascaded Convolutional Networks (MTCNN) for Face Detection and Facial Landmark Alignment. \[online\] Medium. Available at: https://medium.com/@iselagradilla94/multi-task-cascaded-convolutional-networks-mtcnn-for-face-detection-and-facial-landmark-alignment-7c21e8007923 \[Accessed 3 Mar. 2025\].

Jain, A. (2025) Internal working of dlib for plotting facial landmarks. \[online\] Medium. Available at: https://medium.com/@abhishekjainindore24/internal-working-of-dlib-for-plotting-facial-landmarks-b9ba0de4837b \[Accessed 6 Mar. 2025\].

Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J. and Aila, T. (2020) Analyzing and Improving the Image Quality of StyleGAN. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Available at: https://arxiv.org/abs/1912.04958 \[Accessed 28 Jan. 2025\].

Kazemi, V. and Sullivan, J. (2014) One millisecond face alignment with an ensemble of regression trees. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp.1867–1874. Available at: https://ieeexplore.ieee.org/document/6909637 \[Accessed 9 Feb. 2025\].

Kim, D.-K. (2025) SRGAN: The Power of GANs in Super-Resolution. \[online\] Medium. Available at: https://medium.com/@kdk199604/srgan-the-power-of-gans-in-super-resolution-94f39a530a61 \[Accessed 5 Mar. 2025\].

Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., Aitken, A.P., Tejani, A., Totz, J., Wang, Z. and Shi, W. (2017) Photo-realistic single image super-resolution using a generative adversarial network. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Available at: https://ieeexplore.ieee.org/document/8100115 \[Accessed 12 Feb. 2025\].

Luxemburg, R. (2020) 'StyleGAN2 (encoder) \- Official TensorFlow Implementation', *GitHub repository*. Available at: [https://github.com/rolux/stylegan2encoder](https://github.com/rolux/stylegan2encoder) (Accessed: 4 March 2025).

Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G. and Wu, Y. (2018) Mixed precision training. International Conference on Learning Representations (ICLR). Available at: https://arxiv.org/abs/1710.03740 \[Accessed 10 Feb. 2025\].

Qin, X., Tan, X. and Chen, S. (2015) Tri-subject kinship verification: Understanding the core of a family. IEEE Transactions on Multimedia, 17(10), pp.1855–1867. Available at: https://ieeexplore.ieee.org/document/7134765 \[Accessed 26 Jan. 2025\].

Radford, A., Metz, L. and Chintala, S. (2015) Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434. Available at: https://arxiv.org/abs/1511.06434 \[Accessed 15 Feb. 2025\].

Richardson, E., Alaluf, Y., Patashnik, O., Nitzan, Y., Azar, Y. and Shapiro, S. (n.d.) Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation. \[online\] Available at: https://openaccess.thecvf.com/content/CVPR2021/papers/Richardson\_Encoding\_in\_Style\_A\_StyleGAN\_Encoder\_for\_Image-to-Image\_Translation\_CVPR\_2021\_paper.pdf \[Accessed 8 Mar. 2025\].

Shen, Y., Yang, C., Tang, X. and Zhou, B. (2020) InterfaceGAN: Interpreting the latent space of GANs for semantic face editing. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Available at: https://arxiv.org/abs/2005.09635 \[Accessed 3 Mar. 2025\].

Singer, U. (2020) FamilyGan: Generating a Child’s Face using his Parents. \[online\] Medium. Available at: https://medium.com/swlh/familygan-generating-a-childs-face-using-his-parents-394d8face6a4 \[Accessed 6 Mar. 2025\].

Supplementary Material Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation (n.d.) Implementation Details. \[online\] Available at: https://openaccess.thecvf.com/content/CVPR2021/supplemental/Richardson\_Encoding\_in\_Style\_CVPR\_2021\_supplemental.pdf \[Accessed 7 Mar. 2025\].

Srinivas, A., Lin, T., and Abbeel, P. (2022) 'Racial disparity in image generation: An empirical study on the impact of racial composition on generative models'. Available at: https://arxiv.org/pdf/2209.02836 (Accessed: 3 March 2025).​

Visscher, P.M., Hill, W.G. and Wray, N.R. (2008) Heritability in the genomics era—concepts and misconceptions. Nature Reviews Genetics, 9(4), pp.255–266. Available at: https://www.nature.com/articles/nrg2322 \[Accessed 18 Jan. 2025\].

Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., Qiao, Y. and Change Loy, C. (2018) ESRGAN: Enhanced super-resolution generative adversarial networks. Proceedings of the European Conference on Computer Vision (ECCV) Workshops. Available at: https://arxiv.org/abs/1809.00219 \[Accessed 19 Feb. 2025\].

Xie, Y., Wang, H. and Guo, S. (2020) Research on MTCNN face recognition system in low computing power scenarios. 網際網路技術學刊, 21(5), pp.1463–1475. Available at: https://doi.org/10.3966/160792642020092105020 \[Accessed 9 Mar. 2025\].

Zhang, K., Zhang, Z., Li, Z. and Qiao, Y. (2016) Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10), pp.1499–1503. Available at: https://ieeexplore.ieee.org/document/7553523 \[Accessed 5 Feb. 2025\].
