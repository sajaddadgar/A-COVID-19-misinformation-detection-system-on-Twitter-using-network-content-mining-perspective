import re
import emoji
import pandas as pd
from datetime import datetime
import numpy as np
import textstat
from nltk.corpus import stopwords 
from nltk.tokenize import TweetTokenizer, word_tokenize

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import seaborn as sns



pd.set_option('display.max_columns', None)
stop_words = set(stopwords.words('english'))


reliable_users = ['HelenBranswell', 'mlipsitch', 'WHO', 'JeremyFarrar', 'trvrb', 'MackayIM', 'kakape', 'DrTedros', 'cmyeaton', 'CDCDirector', 'T_Inglesby', 'MarionKoopmans', 'edyong209', 'AdamJKucharski', 'neil_ferguson', 'Laurie_Garrett', 'aetiology', 'maiamajumder', 'richardhorton1', 'CEPIvaccines', 'statnews', 'ChristoPhraser', 'arambaut', 'amymaxmen', 'BarackObama', 'marynmck', 'sciencecohen', 'jennifergardy', 'onisillos', 'PeterHotez', 'Chikwe_I', 'DrTomFrieden', 'juliaoftoronto', 'alexandraphelan', 'SCBriand', 'CDCgov', 'mvankerkhove', 'gmleunghku', 'GaviSeth', 'DrMikeRyan', 'michaelmina_lab', 'martinenserink', 'ScottGottliebMD', 'angie_rasmussen', 'CT_Bergstrom', 'CIDRAP', 'simonihay', 'BillGates', 'gatesfoundation', 'gregggonsalves', 'TheLancetInfDis', 'johnbrownstein', 'trevormundel', 'NEJM', 'TheLancet', 'carlzimmer', 'florian_krammer', 'MRC_Outbreak', 'RebeccaKatz5', 'EricTopol', 'devisridhar', 'BogochIsaac', 'nataliexdean', 'BillHanage', 'PeterDaszak', 'LawrenceGostin', 'Atul_Gawande', 'AfricaCDC', 'ProMED_mail', 'CDCGlobal', 'chngin_the_wrld', 'ScienceMagazine', 'wellcometrust', 'gavi', 'JeremyKonyndyk', 'doctorsoumya', 'CarlosdelRio7', 'paimadhu', 'BhadeliaMD', 'edwardcholmes', 'NatureNews', 'LSHTM', 'pathogenomenick', 'JustinLessler', 'lmadoff', 'NIHDirector', 'mugecevik', 'AmeshAA', 'SRileyIDD', 'melindagates', 'healthmap', 'PeterASinger', 'JenniferNuzzo', 'joshmich', 'V2019N', 'phylogenomics', 'ECDC_EU', 'LancetGH', 'InfectiousDz', 'nature', 'c_drosten', 'GlobalFund', 'IlonaKickbusch', 'K_G_Andersen', 'CCDD_HSPH', 'bmj_latest', 'JAMA_current', 'HarvardChanSPH', 'PLOS', 'ashishkjha', 'NPRGoatsandSoda', 'C_Althaus', 'IDEpiPhD', 'OWMorgan', 'WHOAFRO', 'DFisman', 'PardisSabeti', 'Crof', 'KindrachukJason', 'NathanGrubaugh', 'nicolamlow', 'nytimes', 'NickKristof', 'USAIDGH', 'RonaldKlain', 'PATHtweets', 'EpiEllie', 'PLOSPathogens', 'firefoxx66', 'marcelsalathe', 'ASTMH', 'MOUGK', 'dylanbgeorge', 'picardonhealth', 'PLOSMedicine', 'FluTrackers', 'MoetiTshidi', 'DreJoanneLiu', 'JNkengasong', 'yhgrad', 'TheEconomist', 'CDCemergency', 'larrybrilliant', 'profvrr', 'DrNancyM_CDC', 'MSFsci', 'IsabelOtt', 'SaadOmer3', 'EvolveDotZoo', 'TeebzR', 'TAlexPerkins', 'sdwfrost', 'gateshealth', 'LizSzabo', 'SueDHellmann', 'VirusesImmunity', 'mbeisen', 'alexvespi', 'icddr_b', 'glassmanamanda', 'bylenasun', 'SaskiaPopescu', 'KarenGrepin', 'OutbreakJake', 'EpsteinJon', 'jw132', 'DrJayVarma', 'Declan_M_Butler', 'NateSilver538', 'pahowho', 'DoctorYasmin', 'UKAMREnvoy', 'Pezzapezzi', 'RSTMH', 'curefinder', 'gabbystern', 'ChrisJElias', 'EdWhiting1', 'helleringer143', 'megan_b_murray', 'bansallab', 'mpkieny', 'UN', 'UNICEF', 'NIH', 'JeffDSachs', 'MicrobesInfect', 'maggiemfox', 'ghn_news', 'KrutikaKuppalli', 'EckerleIsabella', 'DrJudyStone', 'richardneher', 'eliowa', 'OrinLevine', 'PeterASands', 'bencowling88', 'rozeggo', 'soniashah', 'sarahcobey', 'rd_blueprint', 'BBCBreaking', 'HillaryClinton', 'washingtonpost', 'ChelseaClinton', 'NPRHealth', 'CDCFlu', 'JohnsHopkinsSPH', 'HansRosling', 'NAChristakis', 'womeninGH', 'thelonevirologi', 'BethCameron_DC', 'GlobalBioD', 'cmmid_lshtm', 'EIDGeek', 'igoodfel', 'SunKaiyuan', 'CIDIDteam', 'nmrfaria', 'betzhallo', 'antonioguterres', 'PHE_uk', 'MSF', 'JHSPH_CHS', 'IHME_UW', 'globalhlthtwit', 'HarvardGH', 'VignuzziLab', 'clarewenham', 'epi_michael', 'AOC', 'wef', 'NCDCgov', 'WHOWPRO', 'WHO_Europe', 'syramadad', 'sethmnookin', 'francetim', 'cbpolis', 'DavidQuammen', 'Eurosurveillanc', 'ProfMattFox', 'OlyIlunga', 'stefanswartpet', 'HartlGA', 'petrakle', 'Outbreaks101', 'MorrisonCSIS', 'gail_carson', 'jd_mathbio', 'PFormenty', 'DanielBausch2', 'BBCWorld', 'NYTHealth', 'USAID', 'ASlavitt', 'PIH', 'CMO_England', 'CGDev', 'Craig_A_Spencer', 'matthewherper', 'DrSenait', 'DrLeanaWen', 'GHS', 'DrRichBesser', 'IDSAInfo', 'VirusWhisperer', 'martinmckee', 'davidnabarro', 'sherifink', 'ECDC_Outbreaks', 'RELenski', 'PLOSNTDs', 'CUGHnews', 'EcoHealthNYC', 'arimoin', 'LaurenWeberHP', 'CHGlobalHealth', 'reichlab', 'jocalynclark', 'DokteCoffee', 'svscarpino', '_b_meyer', 'jbloom_lab', 'GermsAndNumbers', 'joel_mossong', 'pam_das', 'lisaschnirring', 'MMKavanagh', 'wtaylor1', 'jLewnard', 'datcummings', 'sciam', 'MaxCRoser', 'laurahelmuth', 'Fogarty_NIH', 'CDDEP', 'FLAHAULT', 'betswrites', 'asoucat', 'AniShakari', 'greg_folkers', 'TomBollyky', 'GYamey', 'jenkatesdc', 'jmayer0716', 'ppenttin', 'sbfnk', 'guardian', 'GretaThunberg', 'newscientist', 'WorldBank', 'guardianscience', 'AcademicsSay', 'royalsociety', 'MicrobiomDigest', 'KulikovUNIATF', 'CDC_NCEZID', 'JATetro', 'baym', 'Wellcome_AMR', 'MartenRobert', 'MCBazacoPhD', 'ZaminIqbal', 'FINDdx', 'DrMartinCDC', 'twpiggott', 'AshTuite', 'neva9257', 'TheMenacheryLab', 'isabelrodbar', 'hayesluk', 'nycbat', 'CJEMetcalf', 'rreithinger', 'kelle569', 'AndyTatem', 'MichelleObama', 'bbchealth', 'WIREDScience', 'MSF_USA', 'PublicHealth', 'CFR_org', 'DFID_UK', 'NatureMedicine', 'choo_ek', 'CDCMMWR', 'CNN', 'DHSCgovuk']

def datetime2milliseconds(date, time):
    return datetime.strptime(date + ' '+ time,'%Y-%m-%d %H:%M:%S').timestamp() * 1000
	
def add_new_col(src_name, dst_name):
    tc = user_df2[src_name]
    tc.reset_index(drop=True, inplace=True)
    tweet_df[dst_name] = tc
	
def join_punctuation(seq, characters='.,;?!'):
    characters = set(characters)
    
    try:
        seq = iter(seq)  
        current = next(seq)
    except StopIteration:
        return
    
    for nxt in seq:
        if nxt in characters:
            current += nxt
        else:
            yield current
            current = nxt
    yield current
	
def preprocessing(text):
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', str(text))
    text = re.sub(r'^RT[\s]+', '', text)
    text = re.sub(r'\@\w*', '', text)
    text = re.sub(r'#', '', text)
    tokenizer = TweetTokenizer(preserve_case=True, strip_handles=True, reduce_len=True)
    text_tokens = tokenizer.tokenize(text)
    return ' '.join(join_punctuation(text_tokens))
	
def calculate_text_statistical_features(text_df):
    reading_ease = [textstat.flesch_reading_ease(doc) for doc in text_df]
    smog_index = [textstat.smog_index(doc) for doc in text_df]
    flesch_kincaid_grade = [textstat.flesch_kincaid_grade(doc) for doc in text_df]
    liau_index = [textstat.coleman_liau_index(doc) for doc in text_df]
    readability_index = [textstat.automated_readability_index(doc) for doc in text_df]
    dale_chall_readability_score = [textstat.dale_chall_readability_score(doc) for doc in text_df]
    difficult_words = [textstat.difficult_words(doc) for doc in text_df]
    linsear_write_formula = [textstat.linsear_write_formula(doc) for doc in text_df]
    gunning_fog = [textstat.gunning_fog(doc) for doc in text_df]
    text_standard = [textstat.text_standard(doc) for doc in text_df]
    return reading_ease, smog_index, flesch_kincaid_grade, liau_index, readability_index, dale_chall_readability_score, difficult_words, linsear_write_formula, gunning_fog, text_standard
	

def extract_emojis(s):
    return ''.join(c for c in s if c in emoji.UNICODE_EMOJI['en'])
	
	
def confusion_matrix_plot(matrix):
    group_counts = ['{0:0.0f}'.format(value) for value in matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in matrix.flatten() / np.sum(matrix)]
    labels = [f'{a}\n{b}' for a, b in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(matrix, annot=labels, fmt='', cmap=sns.light_palette("seagreen", as_cmap=True))
	
	
def evaluate(y_test, prediction):
    print(classification_report(y_test, prediction))

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)

    print('Accuracy score: {}'.format(accuracy))
    print('Precision score: {}'.format(precision))
    print('Recall score: {}'.format(recall))

    return accuracy