function varargout=health_baseline_py(action,varargin)
%HEALTH_BASELINE_PY 隔离的 MATLAB-Python 数值兼容入口。
%
% 本函数是主动 MATLAB 代码中唯一允许直接调用 Python API 的位置，负责：
%   1) 拟合/保存/载入/预测任务书锁定的 scikit-learn HistGB 健康基准；
%   2) 提供 NumPy PCG64 的抽样与状态读写，以逐位复现历史固定结果。
%
% 调用契约：
%   stream = health_baseline_py('make_rng',seed)
%   idx0   = health_baseline_py('randperm',stream,n,k)       % 返回 0 基索引
%   state  = health_baseline_py('capture_rng_state',stream)  % JSON 字符串
%             health_baseline_py('restore_rng_state',stream,state)
%   model  = health_baseline_py('fit_histgb',X,y,seed)
%             health_baseline_py('save_model',model,path)
%   models = health_baseline_py('load_models',paths)
%   tf     = health_baseline_py('is_python_model',model)
%   yhat   = health_baseline_py('predict',model,X)
%
% 输入 X/y 均为 MATLAB double 数组；预测输出为 double 列向量。若本机缺少
% NumPy、scikit-learn 或 joblib，本函数会直接抛出依赖错误，不静默换后端。

action=lower(char(action));
switch action
    case 'make_rng'
        seed=varargin{1};
        np=py.importlib.import_module('numpy');
        varargout{1}=np.random.default_rng(int64(seed));
    case 'randperm'
        stream=varargin{1}; n=varargin{2}; k=varargin{3};
        z=stream.choice(int64(n),int64(k),pyargs('replace',false));
        varargout{1}=double(z);
    case 'capture_rng_state'
        stream=varargin{1};
        json=py.importlib.import_module('json');
        varargout{1}=char(json.dumps(stream.bit_generator.state));
    case 'restore_rng_state'
        stream=varargin{1}; stateJson=varargin{2};
        json=py.importlib.import_module('json');
        stream.bit_generator.state=json.loads(py.str(stateJson));
    case 'fit_histgb'
        X=varargin{1}; y=varargin{2}; seed=varargin{3};
        ens=py.importlib.import_module('sklearn.ensemble');
        np=py.importlib.import_module('numpy');
        model=ens.HistGradientBoostingRegressor(pyargs( ...
            'max_iter',int32(150),'max_depth',int32(6), ...
            'random_state',int32(seed)));
        model.fit(np.asarray(X),np.ravel(np.asarray(y)));
        varargout{1}=model;
    case 'save_model'
        model=varargin{1}; path=varargin{2};
        joblib=py.importlib.import_module('joblib');
        joblib.dump(model,py.str(path));
    case 'load_models'
        paths=varargin{1};
        joblib=py.importlib.import_module('joblib');
        models=cell(size(paths));
        for j=1:numel(paths)
            models{j}=joblib.load(py.str(paths{j}));
        end
        varargout{1}=models;
    case 'is_python_model'
        varargout{1}=startsWith(class(varargin{1}),'py.');
    case 'predict'
        model=varargin{1}; X=varargin{2};
        np=py.importlib.import_module('numpy');
        yhat=double(model.predict(np.asarray(X)));
        varargout{1}=yhat(:);
    otherwise
        error('ncmapss:UnknownPythonAction','未知 Python 兼容动作: %s',action);
end
end
