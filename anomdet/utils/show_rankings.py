from PIL import Image
import numpy as np

def show_rankings(y_true, y_score=None):
    '''
    Input:
        `y_true` - binary labels
        `y_score` - the scores used to rank the samples of X, optional
            if not given, will assume that y_true is already sorted in order from highest ranked to lowest
    '''
    if y_score is not None:
        y_true = y_true[np.argsort(y_score)[::-1]]
    BLOCK_W, BLOCK_H = (10, 10)
    NEG_BLOCK = np.zeros((BLOCK_W, BLOCK_H, 3), 'uint8')
    NEG_BLOCK[2:-2, 2:-2, :] = 100
    POS_BLOCK = NEG_BLOCK.copy()
    POS_BLOCK[:, :, 1] = 0
    POS_BLOCK[:, :, 2] = 0
    
    POS_BLOCK = Image.fromarray(POS_BLOCK)
    NEG_BLOCK = Image.fromarray(NEG_BLOCK)
    
    
    n_samples = len(y_true)
    n_samples_per_row = min(100, n_samples)
    img_width = int(n_samples_per_row * BLOCK_W)
    img_height = int(np.ceil(float(n_samples) / n_samples_per_row) * BLOCK_H)
    whole_img = Image.new('RGB', (img_width, img_height))
    
    for sample_i in range(n_samples):
        
        # Paste into final image
        x_coord = sample_i % n_samples_per_row * BLOCK_W
        y_coord = sample_i / n_samples_per_row * BLOCK_H
        if y_true[sample_i]:
            whole_img.paste(POS_BLOCK, (x_coord, y_coord))
        else:
            whole_img.paste(NEG_BLOCK, (x_coord, y_coord))
    return whole_img

if __name__ == '__main__':
    y_true = np.array([True, True, False, False, False, False, True])
    y_score = np.array([10, 9, 8 ,7 ,6 ,5,11])
    ranking_viz = show_rankings(y_true, y_score)
    ranking_viz.save('viz.png')

    
