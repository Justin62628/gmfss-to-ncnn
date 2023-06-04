# encoding: utf-8
import torch

class LOG_STATE:
    is_log = True
    log_cnt = 0
    max_log_cnt = 100

def print_mat(input_mat: torch.tensor, name):
    if not LOG_STATE.is_log or LOG_STATE.log_cnt >= LOG_STATE.max_log_cnt:
        return
    if len(input_mat.shape) == 3:
        _, h, w = input_mat.shape
        c = 1
        d = 1
    elif len(input_mat.shape) == 4:
        _, c, h, w = input_mat.shape
        d = 1
    elif len(input_mat.shape) == 5:
        _, c, d, h, w = input_mat.shape
    else:
        print(f"escape {name}, {input_mat.shape}")
    with open(f"debug{LOG_STATE.log_cnt}_{w}.{h}.{d}.{c}_{name}.txt", "w", encoding="utf-8") as f:
        f.write(f"shape {w} {h} {d} {c}\n")
        input_mat_dat = input_mat.detach().cpu().numpy()
        for i in range(c):
            for j in range(d):
                for k in range(h):
                    for l in range(w):
                        if len(input_mat.shape) == 3:
                            data = input_mat_dat[0, k, l]
                        elif len(input_mat.shape) == 4:
                            data = input_mat_dat[0, i, k, l]
                        elif len(input_mat.shape) == 5:
                            data = input_mat_dat[0, i, j, k, l]
                        f.write(f"{data:.3f} ")
        print(f"Logging {LOG_STATE.log_cnt}, {name}: w {w}, h {h}, d {d}, c {c}")
        
                        
    LOG_STATE.log_cnt += 1
