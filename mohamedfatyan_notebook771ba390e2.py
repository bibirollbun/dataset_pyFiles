# x_batch: [B, C=3, L] clean sequences
# f_pois, f_clean: loaded models in eval() mode
# positions: iterable of candidate p’s

def splice(x, T, p, mode="replace", lo=None, hi=None):
    x_ = x.clone()
    if mode == "replace":
        x_[:, :, p:p+T.shape[-1]] = T
    else:  # additive
        x_[:, :, p:p+T.shape[-1]] = torch.clamp(
            x_[:, :, p:p+T.shape[-1]] + T, lo, hi
        )
    return x_

def tv_loss(T):
    return (T[:, :, 1:] - T[:, :, :-1]).abs().mean()

def recover_trigger_for_position(p, dataloader, steps=1500, lr=3e-2,
                                 lam_cons=1.0, lam_tv=1e-4, lam_l2=1e-5,
                                 lo=None, hi=None):
    T = torch.zeros(1, 3, 75, requires_grad=True, device=device)
    opt = torch.optim.Adam([T], lr=lr)
    ema_loss = None
    for t in range(steps):
        x = next_batch(dataloader).to(device)  # [B,3,L]
        xT = splice(x, T, p, mode="replace", lo=lo, hi=hi)
        with torch.no_grad():
            y_clean = f_clean(xT)
        y_pois = f_pois(xT)

        discrep = (y_pois - y_clean).pow(2).mean()
        cons = y_pois.var(dim=0).mean()  # encourages collapse to a narrow target
        reg = lam_tv * tv_loss(T) + lam_l2 * (T**2).mean()

        loss = discrep + lam_cons * cons + reg
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            T.data.clamp_(lo, hi)  # keep in sensor range
        # optional: EMA for early stopping
    return T.detach(), final_metrics(...)

