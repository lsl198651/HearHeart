import torch
from evaluation import calculate_murmur_scores
import numpy as np
import utils
classes = ['Present', 'Unknown', 'Absent']


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
	target_all = []
	pred_all = []
	prob_all = []
	murmur_classes = ['Present',  'Absent']#'Unknown',
	loss_avg = utils.RunningAverage()
	labels_ens = np.zeros(1)
	probabilities_ens = np.zeros((2))
	with torch.no_grad():
		for batch_idx, data in enumerate(test_loader):
			inputs = data[0]
			inputs1 = data[1]
			# target = data[1].squeeze(1).to(device)
			# target = data[1].to(device)
			target = data[2].to(device)
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
			target_all.extend(target.cpu().numpy())
			pred_all.append(predicted)
			prob_all.append(prob_ave)
		y_test = np.zeros((total, 2))
		y_pred = np.zeros((total, 2))
		for i in range(total):
			y_test[i, target_all[i]] = 1
			y_pred[i, pred_all[i]] = 1
		score = calculate_murmur_scores(y_test, np.asarray(prob_all), y_pred)
	prob_all = np.asarray(prob_all)
	pred_all = np.asarray(pred_all)
	# if np.all(pred_all == 2):
	#     labels_ens[i] = 2
	#     probabilities_ens[i, :] = np.mean(prob_all, axis=0)
	if np.any(pred_all == 0):
		labels_ens = 0
		probabilities_ens= np.mean(prob_all[np.where(pred_all == 0)[0], :], axis=0)
	else:
		labels_ens= 1
		probabilities_ens = np.mean(prob_all, axis=0)
	
	# voting for the final label, the most voted as label and also the corresponding probability to calculate mean
	voting = np.zeros((2, ))
	voting[0] = np.count_nonzero(labels_ens == 0)
	voting[1] = np.count_nonzero(labels_ens == 1)
    # voting[2] = np.count_nonzero(labels_ens == 2)
	label = np.argmax(voting, axis=0)  # when the count are the same, take as positive or unknown

	prob_murmur = np.mean(probabilities_ens[np.where(labels_ens == label)[0]], axis=0)

	# convert label to one-hot vector
	labels_murmur = np.zeros(len(murmur_classes), dtype=np.int_)
	idx = label
	labels_murmur[idx] = 1

	return loss_avg(),  correct / total, score, y_test, y_pred, prob_all
