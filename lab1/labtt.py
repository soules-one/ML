import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

TEST = pd.read_csv("test.csv")
TRAIN = pd.read_csv("train.csv")


def z_score_1d(x):
    x = np.array(x)
    mn = x.mean()
    st = x.std()
    return (x - mn) / st


def z_score(X):
    mn = np.mean(X, axis=0)
    st = np.std(X, axis=0)
    st[st == 0] = 1e-6

    return (X - mn) / st, [mn, st]


def z_scoreP(X, mn, st):
    st[st == 0] = 1e-6
    return (X - mn) / st


def min_max(X):
    mn = np.min(X, axis=0)
    mx = np.max(X, axis=0)
    r = mx - mn
    r[r == 0] = 1
    return (X - mn) / r, [mn, mx]


def min_maxP(X, mn, mx):
    r = mx - mn
    r[r == 0] = 1
    return (X - mn) / r

def MSE(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    score = np.float64(0)
    if len(y_pred) != len(y_true):
        raise ValueError
    n = len(y_true)
    for i in range(n):
        score += (y_true[i] - y_pred[i]) ** 2
    return score / n

def MAE(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    score = np.float64(0)
    if len(y_pred) != len(y_true):
        raise ValueError
    n = len(y_true)
    for i in range(n):
        score += np.abs(y_true[i] - y_pred[i])
    return score / n

def MAPE(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    score = np.float64(0)
    if len(y_pred) != len(y_true):
        raise ValueError
    n = len(y_true)
    return np.sum(np.abs((y_true - y_pred) / y_true))

def R2(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    score = np.float64(1)
    if len(y_pred) != len(y_true):
        raise ValueError
    n = len(y_true)
    num = np.float64(0)
    den = np.float64(0)
    y_mn = np.mean(y_true)
    for i in range(n):
        num += (y_true[i] - y_pred[i]) ** 2
        den += (y_true[i] - y_mn) ** 2
    if den == 0:
        den = 1
    score -= num / den
    return score

class LinReg():
    class FType:
        An = 1
        GD = 2
        SGD = 3
    class Normalize:
        No_Norm = 0
        Z_Score = 1
        Min_Max = 2

    def __init__(self, f=FType.An, norm=Normalize.No_Norm, eps=1e-10, epochs=None, step=None):
        if (
            f not in [self.FType.An, self.FType.GD, self.FType.SGD] or
            norm not in [self.Normalize.No_Norm, self.Normalize.Min_Max, self.Normalize.Z_Score]
            ):
            raise ValueError
        self.W = None
        self.n = 0
        self.nt = norm
        self.f = f
        self.eps = eps
        self.epochs = epochs
        self.step = step
        self.norm = None
    
    def fit(self, X, y):
        t = np.empty_like(X)
        if X.shape[0] != len(y):
            raise ValueError
        self.n = len(y)
        norm = self.nt
        if norm != self.Normalize.No_Norm:
            if norm == self.Normalize.Z_Score:
                X, self.norm = z_score(X)
            else:
                X, self.norm = min_max(X)
        X = np.hstack((np.ones((self.n, 1)), X))
        m = X.shape[1]
        self.W = np.random.randn(m) * 0.01
        f = self.f
        step = self.step
        eps = self.eps
        epochs = self.epochs
        if f == self.FType.An:
            try:
                self.W = np.linalg.solve(X.T @ X, X.T @ y)
            except np.linalg.LinAlgError:
                Q, R = np.linalg.qr(X, mode='reduced')
                self.W = np.linalg.lstsq(R, Q.T @ y, rcond=None)[0]
        elif f == self.FType.GD:
            if epochs is None:
                epochs = True
            k = 1
            while epochs:
                p = self.W
                curr = X @ self.W
                grad = (1/self.n) * (curr - y) @ X
                self.W -= (1 / k if step is None else step) * grad
                if np.linalg.norm(grad) < eps or np.linalg.norm(self.W - p) < eps:
                    return self
                k += 1
                if epochs is not True:
                    epochs -= 1
            return self
        else:
            if epochs is None:
                epochs = True
            k = 1
            while epochs:
                idx = np.random.randint(0, self.n)
                xi = X[idx, :]
                yi = y[idx]
                p = self.W
                grad = ((xi @ self.W) - yi) * xi
                self.W -= (1 / k if step is None else step) * grad
                if np.linalg.norm(grad) < eps or np.linalg.norm(self.W - p) < eps:
                    return self
                k += 1
                if epochs is not True:
                    epochs -= 1
            return self

    
    def predict(self, X):
        X = np.array(X)
        if self.nt != self.Normalize.No_Norm:
            if self.nt == self.Normalize.Z_Score:
                X = z_scoreP(X, *self.norm)
            else:
                X = min_maxP(X, *self.norm)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        if self.W is None or X.shape[1] != self.W.shape[0]:
            raise ValueError
        return (X @ self.W).flatten()

class LinRegL2():
    class FType:
        An = 1
        GD = 2
        SGD = 3
    class Normalize:
        No_Norm = 0
        Z_Score = 1
        Min_Max = 2

    def __init__(self, f=FType.An, norm=Normalize.No_Norm, eps=1e-10, 
                 epochs=None, step=None, alpha=0.0):
        if (
            f not in [self.FType.An, self.FType.GD, self.FType.SGD] or
            norm not in [self.Normalize.No_Norm, self.Normalize.Min_Max, self.Normalize.Z_Score]
            ):
            raise ValueError
        self.W = None
        self.n = 0
        self.nt = norm
        self.f = f
        self.eps = eps
        self.epochs = epochs
        self.step = step
        self.norm = None
        self.alpha = alpha
    
    def fit(self, X, y):
        t = np.empty_like(X)
        if X.shape[0] != len(y):
            raise ValueError
        self.n = len(y)
        norm = self.nt
        if norm != self.Normalize.No_Norm:
            if norm == self.Normalize.Z_Score:
                X, self.norm = z_score(X)
            else:
                X, self.norm = min_max(X)
        X = np.hstack((np.ones((self.n, 1)), X))
        m = X.shape[1]
        self.W = np.random.randn(m) * 0.01
        f = self.f
        step = self.step
        eps = self.eps
        epochs = self.epochs
        alpha = self.alpha

        if f == self.FType.An:
            try:
                XTX = X.T @ X
                reg_matrix = np.eye(m)
                reg_matrix[0, 0] = 0
                self.W = np.linalg.solve(XTX + alpha * reg_matrix, X.T @ y)
            except np.linalg.LinAlgError:
                XTX = X.T @ X
                reg_matrix = np.eye(m)
                reg_matrix[0, 0] = 0
                A = XTX + alpha * reg_matrix
                Q, R = np.linalg.qr(A, mode='reduced')
                self.W = np.linalg.lstsq(R, Q.T @ (X.T @ y), rcond=None)[0]
        else:
            raise ValueError
    
    def predict(self, X):
        X = np.array(X)
        if self.nt != self.Normalize.No_Norm:
            if self.nt == self.Normalize.Z_Score:
                X = z_scoreP(X, *self.norm)
            else:
                X = min_maxP(X, *self.norm)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        if self.W is None or X.shape[1] != self.W.shape[0]:
            raise ValueError
        return (X @ self.W).flatten()


class LinRegL1(LinReg):
    def __init__(self, lam=0.1, **kw):
        super().__init__(f=self.FType.GD, **kw)
        self.lam = lam

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)

        if self.nt != self.Normalize.No_Norm:
            if self.nt == self.Normalize.Z_Score:
                X, self.norm = z_score(X)
            else:
                X, self.norm = min_max(X)

        self.n = len(y)
        X = np.hstack((np.ones((self.n, 1)), X))
        m = X.shape[1]

        self.W = np.random.randn(m) * 0.01

        epochs = self.epochs if self.epochs is not None else 1000
        step = self.step if self.step is not None else 1e-3

        for k in range(epochs):
            p = self.W.copy()

            curr = X @ self.W
            grad_mse = (1/self.n) * (X.T @ (curr - y))


            grad_l1 = self.lam * np.sign(self.W)
            grad_l1[0] = 0  

            grad = grad_mse + grad_l1
            self.W -= step * grad

            if np.linalg.norm(self.W - p) < self.eps:
                break

        return self
    

class KFold:
    def __init__(self, n, r_state=None):
        if n < 2 or (r_state is not None and (not isinstance(r_state, int) or r_state < 0)):
            raise ValueError
        self.n = n
        if r_state is not None  :
            self.seed = r_state
        else:
            self.seed = np.random.randint(0, 1000000)
    
    def create_splits(self, X, Y):
        if self.n > len(X) or len(X) != len(Y):
            raise ValueError
        splits = []
        perm = np.random.RandomState(seed=self.seed).permutation(np.arange(len(Y)))
        l = int(np.floor(len(X) / self.n))
        for k in range(self.n - 1):
            fold = np.zeros((len(X)), dtype=bool)
            kf = np.zeros((len(X)), dtype=bool)
            fold[perm[k * l: (k + 1) * l]] = True
            kf[perm[:k * l]] = True
            kf[perm[(k + 1) * l: ]] = True
            splits.append([kf, fold])
        fold = np.zeros((len(X)), dtype=bool)
        kf = np.zeros((len(X)), dtype=bool)
        fold[perm[(self.n - 1) * l:]] = True
        kf[perm[:(self.n) * l]] = True
        splits.append([kf, fold])
        return splits, self.seed
    
class LOO:
    def __init__(self, r_state=None):
        if (r_state is not None and (not isinstance(r_state, int) or r_state < 0)):
            raise ValueError
        if r_state is not None:
            self.seed = r_state
        else:
            self.seed = np.random.randint(0, 1000000)

    def create_splits(self, X, Y):
        kf = KFold(len(X), self.seed)
        return kf.create_splits(X, Y)
    
class KValidator:
    def __init__(self, modelcls, params, splitter, F=MSE):
        if not isinstance(splitter, KFold) and not isinstance(splitter, LOO):
            raise ValueError
        self.splitter = splitter
        self.F = F
        self.modelcls = modelcls
        self.params = params
    
    def cross_validate(self, X, Y):
        X = np.array(X)
        Y = np.array(Y, dtype=np.float64)
        splits, _ = self.splitter.create_splits(X, Y)
        bestq = np.inf
        bestp = None
        for param in self.params:
            model = self.modelcls(**param)
            q = np.float64(0)
            for i in splits:
                x_train = X[i[0]]
                y_train = Y[i[0]]
                x_test = X[i[1]]
                y_test = Y[i[1]]
                model.fit(x_train, y_train)
                pred = model.predict(x_test)
                sc = self.F(y_test, pred)
                q += sc
            q = q / len(splits)
            if q < bestq:
                bestq = q
                bestp = param
        return bestq, bestp
    

df = TRAIN
df = df.dropna()  # избавляемся от неполных данных
df = df[df["RiskScore"] <= 100]
sc = np.abs(z_score_1d(df["RiskScore"]))
df = df[sc < 3]

def transform_df(X):
    cols = ["MaritalStatus", "HomeOwnershipStatus", "LoanPurpose", "EmploymentStatus", "EducationLevel"]
    #df["ApplicationYear"] = pd.to_datetime(df["ApplicationDate"]).dt.year.astype(int)
    #df["ApplicationMonth"] = pd.to_datetime(df["ApplicationDate"]).dt.month.astype(int)
    #df["ApplicationQuarter"] = pd.to_datetime(df["ApplicationDate"]).dt.quarter.astype(int)
    #df = df.drop(columns="ApplicationDate")
    X["ApplicationDate"] = pd.to_datetime(X['ApplicationDate']).astype(int)
    if "MaritalStatus" in X.columns:
        X = pd.get_dummies(X, columns=['MaritalStatus'], prefix='Marital', drop_first=True, dtype=int)
    if "HomeOwnershipStatus" in X.columns:
        X = pd.get_dummies(X, columns=['HomeOwnershipStatus'], prefix='OWN', drop_first=True, dtype=int)
    if "LoanPurpose" in X.columns:
        X = pd.get_dummies(X, columns=['LoanPurpose'], prefix='PURP', drop_first=True, dtype=int)
    if "EmploymentStatus" in X.columns:
        X = pd.get_dummies(X, columns=['EmploymentStatus'], prefix='EMP', drop_first=True, dtype=int)
    if "EducationLevel" in X.columns:
        X = pd.get_dummies(X, columns=['EducationLevel'], prefix='EDU', drop_first=True, dtype=int)
    return X.drop(columns=cols, errors="ignore")

df = transform_df(df)

class FeatureSelector:
    def __init__(self):
        self.exclude = []
    
    def fit(self, X: pd.DataFrame, Y, eps=np.float128("1e-100")):
        print(f"Before: {X.shape[1]} features")
        y_ = Y.values
        self.exclude = []
        st = set(X.columns)
        val = KValidator(LinRegL2, [{'norm': LinReg.Normalize.Z_Score, "alpha": 0.6}], KFold(5, 42))
        pscore = None
        wscore = np.inf
        bscore = np.inf
        wf = None
        miss = 0
        cscore = val.cross_validate(X.values, Y.values)[0]
        while pscore is None or (miss < 3 and len(X.columns) > 2) or ( len(X.columns) > 2 and (bscore >= cscore)):
            pscore = cscore
            wscore = np.inf
            wf = None
            for feature in st:
                x_ = X.drop(columns=feature).values
                score = val.cross_validate(x_, y_)[0]
                if score < wscore:
                    wscore = score
                    wf = feature
            cscore = wscore
            print(f"score без {wf} = {wscore}. С - {pscore}")
            self.exclude.append(wf)
            bscore = min(bscore, cscore)
            if bscore < cscore:
                miss += 1
                print(f"miss - {miss}")
            else:
                miss = 0
            st.discard(wf)
            X = X.drop(columns=wf)
        for i in range(miss):
            self.exclude.pop()
        print(f"After: {X.shape[1] + miss} features")

    def transform(self, X: pd.DataFrame):
        return X.drop(columns=self.exclude, errors="ignore")
    
    def save(self, name="excl.txt"):
        with open(name, "w") as file:
            for i in self.exclude:
                file.write(i + '\n')
    
    def load(self, name="excl.txt"):
        try:
            with open(name, "r") as file:
                for line in file:
                    self.exclude.append(line.strip())
            return True
        except:
            return False

class FWFeatureSelector:
    def __init__(self, n_jobs=-1):
        self.exclude = []
        self.n_jobs = n_jobs
    @staticmethod
    def uvr(y_t, y_p):
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - y_t.mean()) ** 2)
        return ss_res / ss_tot if ss_tot > 0 else 0
    
    def _compute_feature_score(self, feature, include, X, Y, folds, uvr_func):
        score = float(0)
        x_ = X[include + [feature]]
        
        for fold in folds:
            xtr, ytr = x_[fold[0]], Y[fold[0]]
            xte, yte = x_[fold[1]], Y[fold[1]]
            
            model = LinReg(norm=LinReg.Normalize.Z_Score)
            model.fit(xtr.values, ytr.values)
            score += uvr_func(yte.values, model.predict(xte.values))
        
        return feature, score / len(folds)
    
    def fit(self, X: pd.DataFrame, Y, eps=np.float128("1e-20")):
        import os
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from functools import partial
        import pandas as pd
        import numpy as np
        
        print(f"Before: {X.shape[1]} features")
        self.exclude = []
        st = set(X.columns)
        folds = KFold(5,42).create_splits(X, Y)[0]
        
        pscore = None
        wscore = np.inf
        bscore = np.inf
        wf = None
        cscore = np.inf
        include = []
        miss = 0
        
        # Определяем число процессов
        n_jobs = self.n_jobs if self.n_jobs > 0 else max(1, os.cpu_count() - 1)
        
        while pscore is None or (miss < 3 and st) or ((pscore >= cscore) and np.abs(pscore - cscore) >= eps and st):
            if cscore > bscore:
                miss += 1
                print(f"miss - {miss}")
            else:
                miss = 0
            
            pscore = cscore
            wscore = np.inf
            wf = None
            
            # Параллельный расчет scores для всех признаков
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Создаем partial функцию с фиксированными аргументами
                compute_func = partial(
                    self._compute_feature_score,
                    include=include,
                    X=X,
                    Y=Y,
                    folds=folds,
                    uvr_func=self.uvr
                )
                
                # Запускаем вычисления параллельно
                future_to_feature = {
                    executor.submit(compute_func, feature): feature 
                    for feature in st
                }
                
                # Собираем результаты по мере готовности
                for future in as_completed(future_to_feature):
                    try:
                        feature, score = future.result()
                        if score < wscore:
                            wscore = score
                            wf = feature
                    except Exception as e:
                        print(f"Ошибка при вычислении признака {future_to_feature[future]}: {e}")
            
            cscore = wscore
            print(f"score c {wf} = {wscore:.6f}. без - {pscore:.6f}")
            
            include.append(wf)
            st.discard(wf)
            bscore = min(cscore, bscore)
        
        for i in range(miss):
            st.add(include.pop())
        
        self.exclude = list(st)
        print(f"After: {len(include)} features")
        return self
    
    def transform(self, X: pd.DataFrame):
        return X.drop(columns=self.exclude, errors="ignore")
    
    def save(self, name="excl.txt"):
        with open(name, "w") as file:
            for i in self.exclude:
                file.write(i + '\n')
    
    def load(self, name="excl.txt"):
        try:
            with open(name, "r") as file:
                self.exclude = [line.strip() for line in file]
            return True
        except:
            return False

class FeatureWeights:
    def __init__(self):
        self.exclude = []
    
    def fit(self, X: pd.DataFrame, Y, eps=np.float128("1e-100"), pp=5000):
        n = X.shape[1]
        print(f"Before: {n} features")
        self.exclude = []
        pscore = None
        bscore = None
        val = KValidator(LinRegL2, [{"norm": LinReg.Normalize.Z_Score}], KFold(5, 42))
        model = LinRegL2(norm=LinReg.Normalize.Z_Score, alpha=10)
        model.fit(X.values[:pp, ], Y.values[:pp])
        miss = 0
        cscore = val.cross_validate(X, Y)[0]
        bscore = cscore
        while pscore is None or (miss < 3 and n > 2) or ( n > 2 and (bscore >= cscore)):
            pscore = cscore
            ww = np.inf
            idx = None
            for i in range(n):
                if np.abs(model.W[i]) < ww:
                    idx = i
                    ww = np.abs(model.W[i])
            self.exclude.append(X.columns[idx])
            n -= 1
            X = X.drop(columns=X.columns[idx])
            model = LinRegL2(norm=LinReg.Normalize.Z_Score, alpha=0.6)
            model.fit(X.values[:pp, ], Y.values[:pp])
            cscore = val.cross_validate(X, Y)[0]
            bscore = min(cscore, bscore)
            if bscore < cscore:
                miss += 1
                print(f"miss - {miss}")
            else:
                miss = 0
            print(f"score без {X.columns[idx]} = {cscore}. С - {pscore}")
        for i in range(miss):
            self.exclude.pop()
        print(f"After: {n + 1} features")

    def transform(self, X: pd.DataFrame):
        return X.drop(columns=self.exclude, errors="ignore")
    
    def save(self, name="excl.txt"):
        with open(name, "w") as file:
            for i in self.exclude:
                file.write(i + '\n')
    
    def load(self, name="excl.txt"):
        try:
            with open(name, "r") as file:
                for line in file:
                    self.exclude.append(line.strip())
            return True
        except:
            return False


def featureExpander(X_O: pd.DataFrame,X: pd.DataFrame, Y, eps=0.1, enable_ratio=False, enable_poly=False):
    from scipy.stats import boxcox
    corrs = X_O.corrwith(Y).abs().sort_values(ascending=False)
    candidates = corrs[corrs > eps].index
    candidates = candidates
    print("Создаю признаки от", candidates)
    d = {}
    for i in range(len(candidates)):
        col = candidates[i]
        if df[col].min() > 0:
            d[f'EX{col}_bc'], lmd = boxcox(X[col].values)
        else:
            d[f'EX{col}_bc'], lmd = boxcox(X[col].values - X[col].min() + 1e-6)
        d[f'EX{col}_cb'] = X[col] ** 3
        d[f'EX{col}_inv'] = 1 / (X[col] + 1e-6)
        d[f'EX{col}_sq'] = X[col] ** 2
        d[f'EX{col}_sqrt'] = np.sqrt(np.abs(X[col]) + 1e-6)
        d[f'EX{col}_log'] = np.log(np.abs(X[col]) + 1e-6)

        if X_O[col].nunique() > 2:
            ln = np.linspace(X_O[col].min(), X_O[col].max(), 4)
            d[f'EX{col}_1'] = (X[col] < ln[1]) + 1
            d[f'EX{col}_2'] = ((X[col] >= ln[1]) & (X[col] < ln[2])) + 1
            d[f'EX{col}_3'] = (X[col] >= ln[3]) + 1
        for j in range(i+1, len(candidates)):
            a, b = col, candidates[j]
            d[f'EX{a}_plus_{b}'] = X[a] + X[b]
            if enable_ratio:
                d[f'EX{a}_div_{b}'] = X[a] / (X[b].abs() + 1e-6)
                d[f'EX{b}_div_{a}'] = X[b] / (X[a].abs() + 1e-6)

            if enable_poly:
                d[f'EX{a}_x_{b}_sqrt'] = np.sqrt(np.abs(X[a] * X[b]) + 1e-6)
                d[f'EX{a}_x_{b}_log'] = np.log(np.abs(X[a] * X[b]) + 1)
                d[f'EX{a}^2_x_{b}^2'] = ((X[a] ** 2) * (X[b] ** 2))
    newdf = pd.DataFrame(d, index=X.index)
    X = pd.concat([X, newdf], axis=1)
    return X

def corr_with_target(X: pd.DataFrame, y, thr=0.05):
    corrs = X.apply(lambda col: np.abs(np.corrcoef(col, y)[0, 1]))
    print(f'less - {list(corrs[corrs <= thr].index)}')
    return corrs[corrs > thr].index

def pair_corr(X: pd.DataFrame, Y, thr=0.5):
    corrs = X.corrwith(Y).abs()
    excl = set()
    for i in range(len(X.columns)):
        col = X.columns[i]
        cc = X.drop(columns=col).corrwith(X[col]).abs()
        for el in cc.index:
            if cc[el] < thr:
                continue
            if el in excl:
                continue
            if corrs[col] < corrs[el]:
                excl.add(col)
                break
            else:
                excl.add(el)
    print(f'excl - {excl}')
    return X.drop(columns=list(excl))


def low_variance(X: pd.DataFrame, thr=1e-5):
    variances = X.var()
    return X[variances[variances > thr].index]

import sklearn.metrics as metrics

target = "RiskScore"
sm = 7000
X = df.drop(columns=target)
Y = df[target]
XO = df.drop(columns=target)
YO = df[target]

val = KValidator(LinReg, [{"norm": LinReg.Normalize.Z_Score,}], KFold(5, 42))

fs = FWFeatureSelector()
print(val.cross_validate(X, Y))
if not fs.load("exRP.txt"):
    fs.fit(X, Y)
    fs.save("exRP.txt")
X = fs.transform(X)
X_ = X
print(val.cross_validate(X, Y))
X = featureExpander(X_, X_, Y, enable_ratio=True, enable_poly=True)
X = X[corr_with_target(X, Y, 0.001)]
print(val.cross_validate(X, Y))
fss = FWFeatureSelector()
if not fss.load("PP.txt"):
    fss.fit(X, Y, 9000)
    fss.save("PP.txt")
print(val.cross_validate(fss.transform(X), Y))
def answer():
    model = LinRegL2(norm=LinReg.Normalize.Z_Score, alpha=0.6)

    XO = df.drop(columns=target)
    YO = df[target]
    X = df.drop(columns=target)
    Y = df[target]
    ffs = FWFeatureSelector()
    ffs.load("exRP.txt")
    X_T = TEST.drop(columns="ID")
    X_T = transform_df(X_T)
    X = ffs.transform(X)
    X_T = ffs.transform(X_T)
    X_T = X_T[corr_with_target(X, Y)]
    X = X[corr_with_target(X, Y)]
    ffs = FWFeatureSelector()
    ffs.load("PP.txt")
    X_T = featureExpander(X, X_T, Y, enable_poly=True, enable_ratio=True)
    X_T = fss.transform(X_T)
    X = featureExpander(X, X, Y, enable_poly=True, enable_ratio=True)
    X = fss.transform(X)
    model.fit(X.values, Y.values)
    Y_T = model.predict(X_T.values)

    odf = pd.DataFrame({"ID": range(5000), target: Y_T})
    print(odf.info())
    odf.to_csv('my_data.csv', index=False) 

answer()
