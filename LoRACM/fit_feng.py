"""Fit each measured cohort AIF to the seven-parameter Feng equation."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from protocol import feng_curve, FENG_NAMES, validate_times


def main(source, output):
    output = Path(output)
    if output.exists() or output.with_suffix('.fits.csv').exists():
        raise FileExistsError('Feng output already exists; choose another path')
    frame = pd.read_csv(source)
    t = validate_times(frame.iloc[:,0].to_numpy(), 'AIF times', allow_zero=True)
    if frame.shape[1] < 3 or len(t) < 8:
        raise ValueError('Cohort fitting needs at least two AIFs and eight time samples')
    rows, fits = [], {'time_min':t}
    for column in frame.columns[1:]:
        y = frame[column].to_numpy(dtype=float)
        if not np.isfinite(y).all() or np.any(y < 0) or y.max() <= 0:
            raise ValueError(f'{column}: AIF must be finite, nonnegative, and nonzero')
        scale = y.max()
        target = y/scale
        # Multistart bounded fitting; rates are positive and delay precedes peak.
        peak = t[np.argmax(y)]
        upper = [np.inf,np.inf,np.inf,100,100,100,max(peak,1e-6)]
        candidates = []
        for rate in (1.,5.,15.):
            guess = [rate*3,.2,.05,rate,.1,.01,min(peak/2,.1)]
            result = least_squares(lambda p:(feng_curve(t,np.asarray(p)[None,:])[0]-target),
                                   guess,bounds=(np.zeros(7),upper),max_nfev=3000)
            candidates.append(result)
        result = min(candidates,key=lambda r:np.sum(r.fun**2))
        if not result.success:
            raise ValueError(f'{column}: Feng fitting did not converge: {result.message}')
        params = result.x.copy(); params[:3] *= scale
        prediction = feng_curve(t,params[None,:])[0]
        if np.any(prediction < -1e-8):
            raise ValueError(f'{column}: fitted Feng curve has negative activity; review this AIF')
        rows.append(dict(zip(FENG_NAMES,params),aif_id=str(column),rmse=float(np.sqrt(np.mean((prediction-y)**2))),
                         relative_rmse=float(np.sqrt(np.mean((prediction-y)**2))/scale)))
        fits[str(column)+'_measured'] = y
        fits[str(column)+'_fitted'] = prediction
        print(f'{column}: relative RMSE {rows[-1]["relative_rmse"]:.4f}',flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output,index=False,mode='x')
    pd.DataFrame(fits).to_csv(output.with_suffix('.fits.csv'),index=False,mode='x')
    print(f'Saved {output}; inspect fit errors and curves before simulation',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    a = p.parse_args(); main(a.input,a.output)
