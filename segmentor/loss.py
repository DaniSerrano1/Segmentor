import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_poly(pred, target, poly_degree):
    """MSE entre los coeficientes polinomiales predichos y los reales."""
    return F.mse_loss(pred[..., :poly_degree + 1], target[..., :poly_degree + 1])

def loss_bounds(pred, target, poly_degree):
    """MSE para los límites de la línea en el eje X."""
    pred_min, pred_max = pred[..., poly_degree + 1], pred[..., poly_degree + 2]
    target_min, target_max = target[..., poly_degree + 1], target[..., poly_degree + 2]
    loss_min = F.mse_loss(pred_min, target_min)
    loss_max = F.mse_loss(pred_max, target_max)
    return loss_min + loss_max

def loss_fit(pred, target, poly_degree):
    """L1 Loss para medir qué tan bien se ajusta la línea a los puntos reales."""
    return F.l1_loss(pred[..., poly_degree + 3:poly_degree + 5], target[..., poly_degree + 3:poly_degree + 5])

def loss_reg(pred):
    """Regularización L2 para estabilidad del modelo."""
    return torch.mean(pred**2)

def loss_conf(pred, target, poly_degree):
    """Entropía cruzada binaria para la confianza."""
    return F.binary_cross_entropy(pred[..., poly_degree + 5], target[..., poly_degree + 5])

def total_loss(pred, target, poly_degree, lambdas):
    """Cálculo de la función de pérdida total, considerando la cuadrícula."""
    l1, l2, l3, l4, l5 = lambdas

    loss_p = loss_poly(pred, target, poly_degree)  # Coeficientes polinomiales
    loss_b = loss_bounds(pred, target, poly_degree)  # Límites x_min y x_max
    loss_f = loss_fit(pred, target, poly_degree)  # Ajuste de la línea a los puntos
    loss_r = loss_reg(pred)  # Regularización
    loss_c = loss_conf(pred, target, poly_degree)  # Confianza

    return l1 * loss_p + l2 * loss_b + l3 * loss_f + l4 * loss_r + l5 * loss_c