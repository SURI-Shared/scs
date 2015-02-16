clear all

disp('------------------------------------------------------------')
disp('WARNING: this can take a very long time to run.')
disp('It may also crash/run out of memory.')
disp('Set run_cvx = false if you just want to run scs.')
disp('------------------------------------------------------------')

save_results = false;
run_cvx = true;
cvx_use_solver = 'sdpt3';
run_scs = true;

ns = [3000,10000,30000];
ms = ceil(ns/2);

density = 0.1;

time_pat_cvx = 'Total CPU time \(secs\)\s*=\s*(?<total>[\d\.]+)';

for i = 1:length(ns)
    seedstr = sprintf('scs_pnorm_ex_%i',i);
    randn('seed',sum(seedstr));rand('seed',sum(seedstr))
    
    n = ns(i);
    m = ms(i);
    pow = pi;
    G = sprandn(m,n,density);
    f = randn(m,1) * n * density;
    
    n=ns(i);
    m=ms(i);
    
    %%
    if run_scs
        %% scs, with power cone formulation
        Anz = nnz(G) + n + 1 + 3*n;
        data.A = sparse([],[],[],m+3*n,2*n+1,Anz);
        data.A(1:m+1,:) = [G sparse(m,n) sparse(m,1); sparse(1,n) ones(1,n) -1];
        for j=1:n
            ej = sparse(n,1);ej(j) = -1;
            data.A(m+1+(j-1)*3+1:m+1+j*3,:) = [sparse(1,n) ej' 0; sparse(1,2*n) -1;  ej' sparse(1,n) 0];
        end
        data.c = [zeros(2*n,1); 1];
        data.b = [f; 0 ; zeros(3*n,1)];
        K = struct('f', m+1, 'p', ones(n,1) / pow);
        
        params.eps = 1e-3;
        
        [out,x_scs,y_scs,s_scs,info] = evalc('scs_direct(data, K, params)');
        out
        if (save_results); save('data/pnorm_scs_direct', 'out'); end
        
        %%
        [out,x_scs,y_scs,s_scs,info] = evalc('scs_indirect(data, K, params)');
        out
        if (save_results); save('data/pnorm_scs_indirect', 'out'); end
        
    end
    %%
    if run_cvx
        try
            tic
            cvx_begin
            cvx_solver(cvx_use_solver)
            variable x(n)
            minimize(norm(x, pow))
            subject to
            G*x == f
            %cvx = evalc('cvx_end')
            cvx_end
            toc
            
        catch err
            cvx.err{i} = err;
        end
        
        if (save_results); save('data/pnorm_cvx', 'cvx'); end;
        
    end
end