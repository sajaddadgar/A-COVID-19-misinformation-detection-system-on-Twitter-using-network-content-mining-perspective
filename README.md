# A COVID-19 misinformation detection system on Twitter using network & content mining perspective
Dataset that have been used in the paper:

| Dataset  | Description | Link |
| ------------- | ------------- | ------------- |
| Checkovid_Claims  | This dataset contains 18,000 tweets collected from fact-checking websites and reliable organizations with expertise relating to the SARS-CoV-2 (Coronavirus) and COVID-19 pandemic, such as the World Health Organization (WHO) divided into INFORMATIVE and MISINFORMATIVE tweets.| [Access the dataset](https://docs.google.com/spreadsheets/d/1VBvjYBXWgZiaKA_judJ3R-8K-Z4oUhTQuKlCr3rU6Ac/edit?usp=sharing) |
| Checkovid_Sentences | For spotting misinformation among facts, We provided a dataset contains 15,636 INFORMATIVE and MISINFORMATIVE sentences about COVID-19. This dataset was developed by dividing each paragraph in our Claims dataset into sentences using the sentence segmentation in the Spacy library and removing pointless and meaningless sentences Manually. | [Access the dataset](https://docs.google.com/spreadsheets/d/17iCCcOq1QfrfDJ0bJF0RLe4vhC5uytOBSL426Tm_v00/edit?usp=sharing) |
| Checkovid_Tweets  | This dataset contains all of the tweet information that is extracted by Twitter API. | [Access the dataset](https://docs.google.com/spreadsheets/d/1b0554t_9rHEWzwkJLAaLmKeE0k9DHej93XiL1aS7NDg/edit?usp=sharing) |
| Checkovid_Users | This dataset contains all of the users information that is extracted by Twitter API. | [Access the dataset](https://docs.google.com/spreadsheets/d/1fTDiNBKvfFmXrIDTHWc5EPfSN6TLvRmRaNWsRA4fLx8/edit?usp=sharing) |
| Checkovid_Twitter | We extracted 43 features from the two above datasets that contain different tweet characteristics and user engagements, such as linguistic features and social context features. | [Access the dataset](https://docs.google.com/spreadsheets/d/1pV6oMV1QT4DawpyULnoMwetm4o3-EAD9eY5nBrtxhMQ/edit?usp=sharing) |

This paper proposed three approaches to detect COVID-19 related misinformation. These approaches are as follow:
* **Network-Based**: [implementation](https://github.com/sajaddadgar/A-COVID-19-misinformation-detection-system-on-Twitter-using-network-content-mining-perspective/blob/main/Network-based/fake%20news%20on%20twitter%20-%20network%20base.ipynb)
* **Content-Based (Similarity models)**: [implementation](https://github.com/sajaddadgar/A-COVID-19-misinformation-detection-system-on-Twitter-using-network-content-mining-perspective/blob/main/Content-based/Similarity%20models/Similarity.ipynb)
* **Content-Based (Text classification models)**: [implementation](https://github.com/sajaddadgar/A-COVID-19-misinformation-detection-system-on-Twitter-using-network-content-mining-perspective/tree/main/Content-based/Text%20classification%20models)


## Checkovid
<img align="center" width="150px" height="150px" src="http://checkovid19.com/static/image/logo1.png">

A COVID-19 fact-checking website

Link: http://checkovid19.com

Source code: [Click](https://github.com/sajaddadgar/Checkovid-version2)


