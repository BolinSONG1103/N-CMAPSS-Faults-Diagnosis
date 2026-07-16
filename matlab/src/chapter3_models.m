classdef chapter3_models
    % 第3章实验用的轻量 MATLAB 神经网络与优化器。
    % 不引入训练框架；只实现任务书需要的 MoE/MLP 两类模型。

    methods (Static)
        function [model,bestVal,history] = train_moe(name,K,mode,learnDir,Hn, ...
                Wtr,THtr,Rtr,Wva,THva,Rva,opts,initStream)
            model = chapter3_models.init_moe(K,mode,learnDir,size(Hn,1),size(Hn,2), ...
                size(Wtr,2),opts.Hidden,initStream);
            if strcmp(mode,'const') && ~learnDir
                model.bias = dlarray(single(mean(Rtr-THtr*Hn.',1).'));
                bestVal = chapter3_models.moe_nrmse(model,Hn,Wva,THva,Rva);
                history = bestVal;
                fprintf('    [%-10s] 无需训练，val NRMSE=%.5f\n',name,bestVal);
                return;
            end

            state = chapter3_models.init_adam_state(model);
            bestVal = inf;
            bestModel = model;
            history = zeros(opts.Epochs,1);
            iter = 0;
            n = size(Wtr,1);
            for ep = 1:opts.Epochs
                order = randperm(initStream,n);
                for first = 1:opts.Batch:n
                    iter = iter+1;
                    idx = order(first:min(first+opts.Batch-1,n));
                    [loss,grads] = dlfeval(@chapter3_models.moe_loss,model,Hn, ...
                        Wtr(idx,:),THtr(idx,:),Rtr(idx,:),opts.DirPenalty); %#ok<ASGLU>
                    [model,state] = chapter3_models.adam_step(model,grads,state,iter,opts.LearnRate);
                end
                v = chapter3_models.moe_nrmse(model,Hn,Wva,THva,Rva);
                history(ep) = v;
                if v<bestVal
                    bestVal = v;
                    bestModel = model;
                end
                if ep==1 || ep==opts.Epochs || mod(ep,max(1,floor(opts.Epochs/5)))==0
                    fprintf('    [%-10s] epoch %3d/%3d  val NRMSE=%.5f\n', ...
                        name,ep,opts.Epochs,v);
                end
            end
            model = bestModel;
            fprintf('    [%-10s] best val NRMSE=%.5f\n',name,bestVal);
        end

        function model = init_moe(K,mode,learnDir,m,n,nIn,hidden,stream)
            invSoftplus1 = single(log(exp(1)-1));
            model = struct('mode',mode,'K',K,'learn_dir',learnDir);
            model.bias = dlarray(zeros(m,1,'single'));
            if strcmp(mode,'moe')
                model.W1 = dlarray(chapter3_models.glorot(hidden,nIn,stream));
                model.b1 = dlarray(zeros(hidden,1,'single'));
                model.W2 = dlarray(chapter3_models.glorot(K,hidden,stream));
                model.b2 = dlarray(zeros(K,1,'single'));
                model.experts = dlarray(invSoftplus1*ones(n,K,'single'));
            elseif strcmp(mode,'free')
                model.W1 = dlarray(chapter3_models.glorot(hidden,nIn,stream));
                model.b1 = dlarray(zeros(hidden,1,'single'));
                model.W2 = dlarray(zeros(n,hidden,'single'));
                model.b2 = dlarray(invSoftplus1*ones(n,1,'single'));
            end
            if learnDir
                model.dH = dlarray(zeros(m,n,'single'));
            end
        end

        function [loss,grads] = moe_loss(model,Hn,W,TH,R,dirPenalty)
            Wd = dlarray(single(W.'));
            THd = dlarray(single(TH.'));
            Rd = dlarray(single(R.'));
            G = chapter3_models.moe_gains_dl(model,Wd);
            Heff = dlarray(single(Hn));
            if model.learn_dir
                Heff = Heff+model.dH;
            end
            pred = Heff*(G.*THd)+model.bias;
            loss = mean((pred-Rd).^2,'all');
            if model.learn_dir
                loss = loss+dirPenalty*sum(model.dH.^2,'all');
            end
            [values,names] = chapter3_models.learnable_cells(model);
            gradValues = dlgradient(loss,values);
            grads = chapter3_models.cells_to_struct(names,gradValues);
        end

        function G = moe_gains_dl(model,Wd)
            nBatch = size(Wd,2);
            if strcmp(model.mode,'const')
                G = ones(9,nBatch,'like',Wd);
            elseif strcmp(model.mode,'free')
                z = model.W2*tanh(model.W1*Wd+model.b1)+model.b2;
                G = chapter3_models.softplus(z);
            else
                logits = model.W2*tanh(model.W1*Wd+model.b1)+model.b2;
                alpha = chapter3_models.softmax_cols(logits);
                experts = chapter3_models.softplus(model.experts); % 9 x K
                G = experts*alpha;
            end
        end

        function G = moe_gains(model,W)
            G = extractdata(chapter3_models.moe_gains_dl(model,dlarray(single(W.')))).';
        end

        function A = moe_alphas(model,W)
            if ~strcmp(model.mode,'moe')
                A = [];
                return;
            end
            Wd = dlarray(single(W.'));
            logits = model.W2*tanh(model.W1*Wd+model.b1)+model.b2;
            A = extractdata(chapter3_models.softmax_cols(logits)).';
        end

        function H = moe_effective_H(model,Hn)
            H = Hn;
            if model.learn_dir
                H = H+double(extractdata(model.dH));
            end
        end

        function v = moe_nrmse(model,Hn,W,TH,R)
            G = chapter3_models.moe_gains(model,W);
            H = chapter3_models.moe_effective_H(model,Hn);
            pred = (G.*TH)*H.'+double(extractdata(model.bias)).';
            v = norm(pred-R,'fro')/max(norm(R,'fro'),1e-12);
        end

        function [model,bestVal,history] = train_ffn(name,kind,sampleFn,valX,valY, ...
                inputDim,opts,initStream)
            model = chapter3_models.init_ffn(kind,inputDim,opts.Hidden,9,initStream);
            state = chapter3_models.init_adam_state(model);
            bestVal = inf;
            bestModel = model;
            history = zeros(opts.Epochs,1);
            iter = 0;
            for ep = 1:opts.Epochs
                [X,Y] = sampleFn();
                n = size(X,1);
                order = randperm(initStream,n);
                for first = 1:opts.Batch:n
                    iter = iter+1;
                    idx = order(first:min(first+opts.Batch-1,n));
                    [~,grads] = dlfeval(@chapter3_models.ffn_loss,model,kind,X(idx,:),Y(idx,:));
                    [model,state] = chapter3_models.adam_step(model,grads,state,iter,opts.LearnRate);
                end
                pred = chapter3_models.predict_ffn(model,kind,valX);
                v = mean((pred-valY).^2,'all');
                history(ep) = v;
                if v<bestVal
                    bestVal = v;
                    bestModel = model;
                end
                if ep==1 || ep==opts.Epochs || mod(ep,max(1,floor(opts.Epochs/4)))==0
                    fprintf('    [%-12s] epoch %3d/%3d  val MSE=%.6f\n', ...
                        name,ep,opts.Epochs,v);
                end
            end
            model = bestModel;
        end

        function model = init_ffn(kind,inputDim,hidden,nOut,stream)
            model = struct();
            model.W1 = dlarray(chapter3_models.glorot(hidden,inputDim,stream));
            model.b1 = dlarray(zeros(hidden,1,'single'));
            if strcmp(kind,'mlp')
                model.W2 = dlarray(chapter3_models.glorot(hidden,hidden,stream));
                model.b2 = dlarray(zeros(hidden,1,'single'));
                model.W3 = dlarray(chapter3_models.glorot(nOut,hidden,stream));
                model.b3 = dlarray(zeros(nOut,1,'single'));
            else
                model.W2 = dlarray(chapter3_models.glorot(nOut,hidden,stream));
                model.b2 = dlarray(zeros(nOut,1,'single'));
            end
        end

        function [loss,grads] = ffn_loss(model,kind,X,Y)
            Xd = dlarray(single(X.'));
            Yd = dlarray(single(Y.'));
            pred = chapter3_models.ffn_forward_dl(model,kind,Xd);
            loss = mean((pred-Yd).^2,'all');
            [values,names] = chapter3_models.learnable_cells(model);
            gradValues = dlgradient(loss,values);
            grads = chapter3_models.cells_to_struct(names,gradValues);
        end

        function pred = predict_ffn(model,kind,X)
            pred = extractdata(chapter3_models.ffn_forward_dl( ...
                model,kind,dlarray(single(X.')))).';
        end

        function y = ffn_forward_dl(model,kind,x)
            if strcmp(kind,'mlp')
                h1 = max(model.W1*x+model.b1,0);
                h2 = max(model.W2*h1+model.b2,0);
                y = model.W3*h2+model.b3;
            else
                h = chapter3_models.gelu(model.W1*x+model.b1);
                y = model.W2*h+model.b2;
            end
        end

        function n = parameter_count(model)
            n = 0;
            fields = fieldnames(model);
            for i = 1:numel(fields)
                value = model.(fields{i});
                if isa(value,'dlarray')
                    n = n+numel(value);
                end
            end
        end

        function state = init_adam_state(params)
            state = struct('m',struct(),'v',struct());
            fields = fieldnames(params);
            for i = 1:numel(fields)
                f = fields{i};
                if isa(params.(f),'dlarray')
                    state.m.(f) = zeros(size(params.(f)),'single');
                    state.v.(f) = zeros(size(params.(f)),'single');
                end
            end
        end

        function [values,names] = learnable_cells(params)
            fields = fieldnames(params);
            names = {};
            values = {};
            for i = 1:numel(fields)
                f = fields{i};
                if isa(params.(f),'dlarray')
                    names{end+1} = f; %#ok<AGROW>
                    values{end+1} = params.(f); %#ok<AGROW>
                end
            end
        end

        function s = cells_to_struct(names,values)
            s = struct();
            for i = 1:numel(names)
                s.(names{i}) = values{i};
            end
        end

        function [params,state] = adam_step(params,grads,state,iter,lr)
            beta1 = 0.9; beta2 = 0.999; eps0 = 1e-8;
            fields = fieldnames(params);
            for i = 1:numel(fields)
                f = fields{i};
                if ~isa(params.(f),'dlarray') || ~isfield(grads,f)
                    continue;
                end
                g = single(extractdata(grads.(f)));
                p = single(extractdata(params.(f)));
                state.m.(f) = beta1*state.m.(f)+(1-beta1)*g;
                state.v.(f) = beta2*state.v.(f)+(1-beta2)*(g.^2);
                mh = state.m.(f)/(1-beta1^iter);
                vh = state.v.(f)/(1-beta2^iter);
                params.(f) = dlarray(p-lr*mh./(sqrt(vh)+eps0));
            end
        end

        function x = glorot(nOut,nIn,stream)
            lim = sqrt(6/(nIn+nOut));
            x = single((2*rand(stream,nOut,nIn)-1)*lim);
        end

        function y = softplus(x)
            y = max(x,0)+log(1+exp(-abs(x)));
        end

        function y = softmax_cols(x)
            z = x-max(x,[],1);
            e = exp(z);
            y = e./sum(e,1);
        end

        function y = gelu(x)
            y = 0.5*x.*(1+tanh(sqrt(2/pi)*(x+0.044715*x.^3)));
        end
    end
end
