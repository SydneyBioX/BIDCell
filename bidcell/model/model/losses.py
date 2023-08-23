import torch
import torch.nn as nn


class NucleiEncapsulationLoss(nn.Module):
    """
    Ensure that nuclei are fully within predicted cells
    """

    def __init__(self, weight, device) -> None:
        super(NucleiEncapsulationLoss, self).__init__()
        self.weight = weight
        self.device = device

    def forward(self, seg_pred, batch_n):
        criterion_ce = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion_ce(seg_pred, batch_n[:, 0, :, :])

        return self.weight * loss


class Oversegmentation(nn.Module):
    """
    Minimise oversegmentation
    """

    def __init__(self, weight, device) -> None:
        super(Oversegmentation, self).__init__()
        self.weight = weight
        self.device = device

    def forward(self, seg_pred, batch_n):
        batch_n = batch_n[:, 0, :, :]

        seg_probs = torch.nn.functional.softmax(seg_pred, dim=1)
        probs_nuc = seg_probs[:, 1, :, :] * batch_n

        mask_cyto = torch.ones(batch_n.shape).to(self.device) - batch_n
        probs_cyto = seg_probs[:, 1, :, :] * mask_cyto

        ones = torch.ones(probs_cyto.shape).to(self.device)
        zeros = torch.zeros(probs_cyto.shape).to(self.device)

        alpha = 1.0

        preds_nuc = torch.sigmoid((probs_nuc - 0.5) * alpha) * (ones - zeros) + zeros
        count_nuc = torch.sum(preds_nuc)

        preds_cyto = torch.sigmoid((probs_cyto - 0.5) * alpha) * (ones - zeros) + zeros
        count_cyto = torch.sum(preds_cyto)

        extra = count_cyto - count_nuc
        m = torch.nn.ReLU()
        loss = m(extra)

        loss = loss / seg_pred.shape[0]

        return self.weight * loss


class CellCallingLoss(nn.Module):
    """
    Maximise assignment of transcripts to cells
    """

    def __init__(self, weight, device) -> None:
        super(CellCallingLoss, self).__init__()
        self.weight = weight
        self.device = device

    def forward(self, seg_pred, batch_sa):
        # Limit to searchable area where there is detected expression
        penalisable = batch_sa * 1
        criterion_ce = torch.nn.CrossEntropyLoss(reduction="none")
        loss = criterion_ce(seg_pred, penalisable[:, 0, :, :])

        loss_total = torch.sum(loss)

        loss_total = loss_total / seg_pred.shape[0]

        return self.weight * loss_total


class OverlapLoss(nn.Module):
    """
    Penalise overlaps between different cells
    """

    def __init__(self, weight, device) -> None:
        super(OverlapLoss, self).__init__()
        self.weight = weight
        self.device = device

    def forward(self, seg_pred, batch_n):
        batch_n = batch_n[:, 0, :, :]
        seg_probs = torch.nn.functional.softmax(seg_pred, dim=1)

        all_nuclei = torch.sum(batch_n, 0)
        all_not_nuclei = torch.ones(batch_n.shape).to(self.device) - all_nuclei

        probs_cyto = seg_probs[:, 1, :, :] * all_not_nuclei

        ones = torch.ones(probs_cyto.shape).to(self.device)
        zeros = torch.zeros(probs_cyto.shape).to(self.device)

        # Penalise if number of cell prob > 0.5 is > 1
        alpha = 1.0
        preds_cyto = torch.sigmoid((probs_cyto - 0.5) * alpha) * (ones - zeros) + zeros
        count_cyto_overlap = torch.sum(preds_cyto, 0) - all_not_nuclei
        m = torch.nn.ReLU()
        count_cyto_overlap = m(count_cyto_overlap)

        loss = torch.sum(count_cyto_overlap)

        scale = seg_pred.shape[0] * seg_pred.shape[2] * seg_pred.shape[3]
        loss = loss / scale

        return self.weight * loss


class PosNegMarkerLoss(nn.Module):
    """
    Positive and negative markers of cell type
    """

    def __init__(self, weight_pos, weight_neg, device) -> None:
        super(PosNegMarkerLoss, self).__init__()
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        self.device = device

    def forward(self, seg_pred, batch_pos, batch_neg):
        batch_pos = batch_pos[:, 0, :, :]
        batch_neg = batch_neg[:, 0, :, :]

        # POSITIVE markers
        criterion_ce = torch.nn.CrossEntropyLoss(reduction="sum")
        loss_pos = criterion_ce(seg_pred, batch_pos)

        # NEGATIVE markers
        seg_probs = torch.nn.functional.softmax(seg_pred, dim=1)
        probs_cells = seg_probs[:, 1, :, :]

        ones = torch.ones(probs_cells.shape).to(self.device)
        zeros = torch.zeros(probs_cells.shape).to(self.device)

        alpha = 1.0

        preds_cells = (
            torch.sigmoid((probs_cells - 0.5) * alpha) * (ones - zeros) + zeros
        )

        loss_neg = torch.sum(preds_cells * batch_neg)

        loss_total = (
            self.weight_pos * loss_pos + self.weight_neg * loss_neg
        ) / seg_pred.shape[0]

        return loss_total
