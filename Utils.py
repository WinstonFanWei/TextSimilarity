import os
import logging
from sklearn.metrics import f1_score

def rmse(predictions, targets):
    return (((predictions - targets) ** 2).mean()) ** 0.5

def compute_rmse(compare_result, paras):
    
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler('log/debug.log'),
                        logging.StreamHandler()
                    ])
    logger = logging.getLogger(__name__)
    
    if paras["MySimilarityCompute"] == True:
        rmse_my = round(rmse(compare_result["Similarity"], compare_result["mySimilarity"]), 4)
        # print("[mySimilarity] RMSE: ", rmse_my)
    else:
        rmse_my = None
        # print("[mySimilarity] CORR: ", rmse_my)
    rmse_cosine = round(rmse(compare_result["Similarity"], compare_result["Similarity_cosine"]), 4)
    # print("[Similarity_cosine] RMSE: ", rmse_cosine)
    rmse_doc_topic = round(rmse(compare_result["Similarity"], compare_result["Similarity_doc_topic"]), 4)
    # print("[Similarity_doc_topic] RMSE: ", rmse_doc_topic)
    rmse_half = round(rmse(compare_result["Similarity"], compare_result["Similarity_sentence_topic"]), 4)
    # print("[Similarity_sentence_topic] RMSE: ", rmse_half)
    
    message = "\n---------------------- RMSE ---------------------- \n[0, +inf] RMSE smaller is better.\n[mySimilarity] RMSE: " + str(rmse_my) + "\n[Similarity_cosine] RMSE: " + str(rmse_cosine) + "\n[Similarity_doc_topic] RMSE: " + str(rmse_doc_topic) + "\n[Similarity_sentence_topic] RMSE: " + str(rmse_half) + "\n--------------------------------------------------\n"
    logger.debug(message)
    
def compute_correlation(compare_result, paras):
    
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler('log/debug.log'),
                        logging.StreamHandler()
                    ])
    logger = logging.getLogger(__name__)
    
    if paras["MySimilarityCompute"] == True:
        cor_my = round(compare_result["Similarity"].corr(compare_result["mySimilarity"]), 4)
        # print("[mySimilarity] CORR: ", cor_my)
    else:
        cor_my = None
        # print("[mySimilarity] CORR: ", cor_my)
    cor_cosine = round(compare_result["Similarity"].corr(compare_result["Similarity_cosine"]), 4)
    # print("[Similarity_cosine] CORR: ", cor_cosine)
    cor_doc_topic = round(compare_result["Similarity"].corr(compare_result["Similarity_doc_topic"]), 4)
    # print("[Similarity_doc_topic] CORR: ", cor_doc_topic)
    cor_half = round(compare_result["Similarity"].corr(compare_result["Similarity_sentence_topic"]), 4)
    # print("[Similarity_sentence_topic] CORR: ", cor_half)
    
    message = "\n---------------------- Correlation ---------------------- \n[-1, 1] Correlation bigger is better.\n[mySimilarity] CORR: " + str(cor_my) + "\n[Similarity_cosine] CORR: " + str(cor_cosine) + "\n[Similarity_doc_topic] CORR: " + str(cor_doc_topic) + "\n[Similarity_sentence_topic] CORR: " + str(cor_half) + "\n---------------------------------------------------------\n"
    logger.debug(message)
    
def df_add_labels(df, column1, column2):
    df["ground_truth"] = (df[str(column1)] > 0.5).astype(int)
    df["predict_label"] = (df[str(column2)] > 0.5).astype(int)
    return df

def compute_f1(compare_result, paras):
    
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler('log/debug.log'),
                        logging.StreamHandler()
                    ])
    logger = logging.getLogger(__name__)
    
    if paras["MySimilarityCompute"] == True:
        df = df_add_labels(compare_result, "Similarity", "mySimilarity")
        f1_my = round(f1_score(df['ground_truth'], df['predict_label']), 4)
        # print("[mySimilarity] F1-score: ", f1_my)
    else:
        f1_my = None
        # print("[mySimilarity] F1-score: ", f1_my)
        
    df = df_add_labels(compare_result, "Similarity", "Similarity_cosine")
    f1_cosine = round(f1_score(df['ground_truth'], df['predict_label']), 4)
    # print("[Similarity_cosine] F1-score: ", f1_cosine)
    df = df_add_labels(compare_result, "Similarity", "Similarity_doc_topic")
    f1_doc_topic = round(f1_score(df['ground_truth'], df['predict_label']), 4)
    # print("[Similarity_doc_topic] F1-score: ", f1_doc_topic)
    df = df_add_labels(compare_result, "Similarity", "Similarity_sentence_topic")
    f1_half = round(f1_score(df['ground_truth'], df['predict_label']), 4)
    # print("[Similarity_sentence_topic] F1-score: ", f1_half)
    
    message = "\n---------------------- F1-score ---------------------- \n[0, 1] F1-score bigger is better.\n[mySimilarity] F1-score: " + str(f1_my) + "\n[Similarity_cosine] F1-score: " + str(f1_cosine) + "\n[Similarity_doc_topic] F1-score: " + str(f1_doc_topic) + "\n[Similarity_sentence_topic] F1-score: " + str(f1_half) + "\n------------------------------------------------------\n"
    logger.debug(message)
    
    