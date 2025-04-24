import torch

from utils import compute_scale_and_shift, compute_errors, compute_metrics, RunningAverageDict

def evaluate(preds, gts, sport_name, device, mask_need=False):
    mask = torch.ones(*preds.shape, device=device)
    mask_score = False

    if sport_name == "basketball":
        mask[870:1016, 1570:1829] = 0
    elif sport_name == "football":
        raise NotImplementedError

    # with open('test_score.txt', 'r') as f:
    #     file_contents = f.read().splitlines()

    mask.to(device)

    # i=0
    with torch.no_grad():
        # for preds, gts in zip(pred_loader, gt_loader):
        mask_score = False
        preds, gts = preds.to(device), gts.to(device)

        # Special case for some of the soccer test files that contain a score banner
        # gt_file = gt_files[i]
        # if gt_file in file_contents:
        #     mask[70:122, 95:612] = 0
        #     mask_score = True
        # i+=1

        # gts = gts / 65535.0
        # if torch.all(gts == 1) or torch.all(gts == 0):
        #     continue
        # preds = preds / 65535.0

        scale, shift = compute_scale_and_shift(preds, gts, mask)
        scaled_predictions = scale.view(-1, 1, 1) * preds + shift.view(-1, 1, 1)
        return compute_metrics(gts, scaled_predictions[0], mask_score, sport_name, mask_needed=mask_need)
