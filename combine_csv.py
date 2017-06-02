dir = "/mnt/home/dunan/Learn/Tensorflow/youtbe-8m/"

gap80 = ["prediction_PeeholeLSTM_1.csv",
         "predictions_moe16.csv"]

gap79 = ["predictions_eval_final.csv",
         "predictions_moe_25.csv",
         "predictions_25_all_train_val.csv",
         "preiction_RandomLSTM_2.csv",
         "preiction_RandomLSTM_3.csv",
         "prediction_LayerNormLstmModel_1.csv",
         "prediction_BiLstm_1.csv",
         "prediction_SeqCNNLstmModel_1.csv"]

gap78 = ["predictions_3nnBnReluDropSkip_0_79.csv",
         "prediction_4LayerLstm_1.csv",
         "predictions_dbof_average.csv",
         "prediction_Grid2LSTM_1.csv",
         "predictions_dbof.csv",
         "preiction_RandomLSTM_1.csv",
         "result_mlp2.csv",
         "predictions_30_all_train_val_15010.csv"]

gap77 = ["predictions_3nnBnReluDropSkip_0_78.csv"]

sum_weight = 0.2*len(gap80) + 0.15*len(gap79) + 0.1*len(gap78)
weight_dict = {}
for file in gap80:
    weight_dict[file] = 0.2/sum_weight

for file in gap79:
    weight_dict[file] = 0.15/sum_weight

for file in gap78:
    weight_dict[file] = 0.1/sum_weight

file_list = weight_dict.keys()
num_file = len(file_list)

output = "/mnt/home/dunan/Learn/Tensorflow/youtbe-8m/20170530_new_weighted_17_model.csv"
id_dict = {}
for index, file in enumerate(file_list):
    path = dir + file
    with open(path) as f:
        if index == 0:
            for line in f:
                if line[0] != "V":
                    sp = line.strip().split(",")
                    video_id = sp[0]
                    video_confidence = {}
                    conf_sp = sp[1].split(" ")
                    # print conf_sp
                    for i in range(len(conf_sp)):
                        if i % 2 != 0:
                            video_confidence[conf_sp[i - 1]] = float(conf_sp[i]) * weight_dict[file]

                    id_dict[video_id] = video_confidence

        elif index == num_file - 1:

            with open(output, "w") as fout:
                print >> fout, "VideoId,LabelConfidencePairs"
                for line in f:
                    if line[0] != "V":
                        sp = line.strip().split(",")
                        video_id = sp[0]
                        video_confidence = id_dict[video_id]
                        conf_sp = sp[1].split(" ")
                        score_pair = []
                        processed = {}
                        for i in range(len(conf_sp)):
                            if i % 2 != 0:
                                tag_id = conf_sp[i - 1]
                                tag_score = video_confidence.get(tag_id, 0.0)
                                score = float(conf_sp[i])
                                new_score = tag_score + score * weight_dict[file]
                                # new_score = tag_score**(0.7)*score**(0.3)
                                score_pair.append((new_score, tag_id))
                                processed[tag_id] = True

                        for tag_id in video_confidence.keys():
                            if processed.get(tag_id, False) is False:
                                score_pair.append((video_confidence[tag_id], tag_id))

                        score_pair.sort(reverse=True)
                        # print score_pair
                        conf_table = []
                        for i in range(20):
                            conf_table.append(str(score_pair[i][1]))
                            conf_table.append(format(score_pair[i][0], "0.6f"))
                            
                        print >> fout, ",".join([video_id, " ".join(conf_table)])

        else:
            for line in f:
                if line[0] != "V":
                    sp = line.strip().split(",")
                    video_id = sp[0]
                    video_confidence = id_dict[video_id]
                    conf_sp = sp[1].split(" ")
                    score_pair = []
                    processed = {}
                    for i in range(len(conf_sp)):
                        if i % 2 != 0:
                            tag_id = conf_sp[i - 1]
                            tag_score = video_confidence.get(tag_id, 0.0)
                            score = float(conf_sp[i])
                            new_score = tag_score + score * weight_dict[file]
                            # new_score = tag_score**(0.7)*score**(0.3)
                            video_confidence[tag_id] = new_score
                            processed[tag_id] = True