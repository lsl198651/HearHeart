import torch
from evaluation import calculate_murmur_scores
import numpy as np
import utils
from evaluate_model import compute_auc, compute_accuracy
from torcheval.metrics.functional import binary_auprc, binary_auroc,binary_accuracy,binary_f1_score,binary_confusion_matrix
classes = ['Present',  'Absent']


def evaluate(model, device, test_loader, loss_fn):
	correct = 0
	total = 0
	model.eval()
	target_all = []
	pred_all = []
	prob_all = []
	loss_avg = utils.RunningAverage()
	with torch.no_grad():
		for batch_idx, data in enumerate(test_loader):
			inputs = data[0].to(device)
			# target = data[1].squeeze(1).to(device)
			target = data[1].to(device)

			outputs = model(inputs)

			loss = loss_fn(outputs, target)
			loss_avg.update(loss.item())
			_, predicted = torch.max(outputs.data, 1)
			total += target.size(0)
			correct += (predicted == target).sum().item()
			target_all.extend(target.cpu().numpy())
			pred_all.extend(predicted.cpu().numpy())
			prob_temp = torch.softmax(outputs, dim=1)
			prob_all.extend(prob_temp.data.cpu().numpy())
		y_test = np.zeros((total, 3))
		y_pred = np.zeros((total, 3))
		for i in range(total):
			y_test[i, target_all[i]] = 1
			y_pred[i, pred_all[i]] = 1
		score = calculate_murmur_scores(y_test, prob_all, y_pred)
	return loss_avg(), 100*correct/total, score, y_test, y_pred, prob_all


def evaluate_patch(model, device, test_loader, loss_fn):
	correct = 0
	total = 0
	model.eval()
	target_seg=[]
	target_all = []
	labels_all=[]
	pred_all = []
	prob_all = []
	labels_murmur_all=[]
	patient_pred={}
	patient_prob={}
	patient_target={}
	murmur_classes = ['Present',  'Absent']#'Unknown',
	loss_avg = utils.RunningAverage()
	labels_ens = np.zeros(1)
	probabilities_ens = np.zeros((2))
	with torch.no_grad():
		for batch_idx, data in enumerate(test_loader):
			inputs = data[0]#[1,1,128,601]
			inputs1 = data[1]
			# target = data[1].squeeze(1).to(device)
			# target = data[1].to(device)
			target = data[2].to(device)
			patient_id=data[3]
			if not patient_id in patient_pred.keys():
				patient_pred[patient_id]=[]
				patient_prob[patient_id]=[]
			else:
				pass
			if not patient_id in patient_target.keys():
				patient_target[patient_id]=target
			else:
				pass
			prob = []
			for idx, input in enumerate(inputs):
				outputs = model(input.to(device), inputs1[idx].to(device))
				loss = loss_fn(outputs, target)
				loss_avg.update(loss.item())
				# _, predicted = torch.max(outputs.data, 1)
				prob_temp = torch.softmax(outputs, dim=1)  # first use the average possibility of patches to classify
				prob.extend(prob_temp.data.cpu().numpy())
			prob_ave = np.mean(np.asarray(prob), axis=0)
			predicted = np.argmax(prob_ave)
			total += target.size(0)
			correct += (predicted == target).sum().item()
			target_seg.extend(target.cpu().numpy())
			pred_all.append(predicted)#预测出来的标签
			prob_all.append(prob_ave)#0和1的概率
			patient_pred[patient_id].append(predicted)
			patient_prob[patient_id].append(prob_ave)
		
		for id,preds in patient_pred.items():			
			prob_all_arr = np.asarray(patient_prob[id])
			pred_all_arr = np.asarray(preds)
			target_p=patient_target[id]
			if np.any(pred_all_arr == 0):
				labels_ens = 0
				probabilities_ens= np.mean(prob_all_arr[np.where(pred_all_arr == 0)[0], :], axis=0)
			else:
				labels_ens= 1
				probabilities_ens = np.mean(prob_all_arr, axis=0)
			
			# voting for the final label, the most voted as label and also the corresponding probability to calculate mean
			voting = np.zeros((2, ))
			voting[0] = np.count_nonzero(labels_ens == 0)
			voting[1] = np.count_nonzero(labels_ens == 1)
			# voting[2] = np.count_nonzero(labels_ens == 2)
			label = np.argmax(voting, axis=0)  # when the count are the same, take as positive or unknown
			target_all.append(target_p)
			labels_all.append(label)
	labels_patients,target_patients=torch.tensor(labels_all),torch.tensor(target_all),
	prc=binary_auprc(labels_patients,target_patients)
	roc=binary_auroc(labels_patients,target_patients)
	acc=binary_accuracy(labels_patients,target_patients)
	f1=binary_f1_score(labels_patients,target_patients)
	cm=binary_confusion_matrix(labels_patients,target_patients)
	# labels_segm,target_segm=torch.tensor(pred_all),torch.tensor(target_seg),
	# prc_seg=binary_auprc(labels_segm,target_segm)
	# roc_seg=binary_auroc(labels_segm,target_segm)
	# acc_seg=binary_accuracy(labels_segm,target_segm)
	# f1_seg=binary_f1_score(labels_segm,target_segm)
	# confusion_matrix=binary_confusion_matrix(labels_segm,target_segm)
	# print(f'----segments_wise---- \n acc={acc_seg:.3%}\n roc:{roc_seg:.3f}\n prc:{prc_seg:.3f}\n f1:{f1_seg:.3f}')
	# print(confusion_matrix)
	# 现在还缺一个segments-wise的性能指标
	return loss_avg(),acc,roc,prc,f1,cm
