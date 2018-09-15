function [seg_] = normcut_temp(img, img_file_path)
    
    fprintf('** Progress the image segmentation for "%s" is started. **\n\n', img_file_path);

    file = split(img_file_path, ".");
    file1 = char(strcat(file(1), '_affinity.mat'));
    file2 = char(strcat(file(1), '_segmentation.mat'));

    M = 16; % patch size (M-by-M)
    m = 12; % window (within patch) size (m-by-m)
    K = 10; % number of segments

    img_lab = rgb2lab(img);
    img_lab = padarray(img_lab, [8 8]);

    [height, width, ~] = size(img_lab);

    function [mu, lik, s] = affprop(L, K, T)
        % total constraint messages
        function S_out = f2vmarkov_sumMPK(S, K)
            I = length(S);
            [Ssort, Sorder] = sort(S);
            S_out = zeros(1, I);
            S_out(Sorder(end - K + 1 : end)) = -Ssort(end - K);
            S_out(Sorder(1 : end - K)) = -Ssort(end - K + 1);
            return;
        end

        % self-explanatory helper functions
        function arg = argmin(varargin)
            [~, arg] = min(varargin{:});
            return;
        end

        function arg = argmax(varargin)
            [~, arg] = max(varargin{:});
            return;
        end

        function arg = argsort(varargin)
            [~, arg] = sort(varargin{:});
            return;
        end

        % input parameter checking
        if nargin < 3 || numel(T) ~= 1 || T < 0 
            T = 40;
        end
        
        if nargin < 2
            error('This function requires a minimum of two input arguments');
        end
        
        [I1, I2, I3] = size(L);
        
        if (I1 ~= I2 || I3 ~= 1)
            error('Affinity matrix L must be square');
        else
            I = I1;
        end
        
        if numel(K) > 1
            K = round(sum(K .* (0 : length(K) - 1)'));
        end % E[K] if distribution

        % initialize internal variables
        diag = sub2ind([I I], 1 : I, 1 : I);
        notdiag = [];
        for i = 1 : I
            notdiag = [notdiag; sub2ind([I I], repmat(i, [1 I - 1]), [1 : i - 1 i + 1 : I])];
        end
        
        r = zeros(I, I); % responsibilities r_ik: ln P(i in k)/maxP(i in j)
        a = zeros(I, I); % availabilities a_ik: ln P(k is i's center)/P(k isn't i's center)
        tau = zeros(1, I); % total constraint messages: ln P(k is center)/P(k isn't center)
        
        rin = L; % I-by-I matrix of A + L + tau
        rinbest = argmax(rin, [], 2);
        temp = rin;
        temp(sub2ind([I I], (1 : I)', rinbest)) = -Inf;
        rinnextbest = argmax(temp, [], 2);
        clear temp;
        
        for t = 1 : T % affinity propagation iterations
            for k = argsort(-sum(L)) % looping through training cases
                % update r
                bestnotk = rinbest;
                temp = find(bestnotk == k);
                bestnotk(temp) = rinnextbest(temp);
                r(:, k) = L(:, k) - rin(sub2ind([I I], (1 : I)', bestnotk));
                r(k, k) = r(k, k) + tau(k);

                % update a
                max0rk = max(0, r(:, k));
                a1_k = sum(max0rk) - max0rk - max0rk(k) + r(k, k);
                a1_k(k) = a1_k(k) + max0rk(k) + r(k, k);
                rk = r(:, k);
                rk(k) = -inf;
                [maxrknoti, argmaxrknoti] = max(rk);
                rk(argmaxrknoti) = -inf;
                maxrknoti = repmat(maxrknoti, [I 1]);
                maxrknoti(argmaxrknoti) = max(rk);
                a2_k = max(0, -maxrknoti);
                a(:, k) = min(a1_k, a2_k);
                a(k, k) = min(0, max(r([1 : k - 1 k + 1 : I], k))) + sum(max(0, r([1 : k - 1 k + 1 : I], k))); % clean up a(k,k)

                % update rin
                rin(:, k) = a(:, k) + L(:, k);
                rin(k, k) = rin(k, k) + tau(k);
                rinbest = argmax(rin, [], 2);
                temp = rin;
                temp(sub2ind([I I], (1 : I)', rinbest)) = -Inf;
                rinnextbest = argmax(temp, [], 2);
                clear temp;
            end
            
            tau = f2vmarkov_sumMPK(a(diag) + L(diag) - max(a(notdiag) + L(notdiag), [], 2)', K);
            rin = a + L;
            rin(diag) = rin(diag) + tau;
            [~, rinbest] = max(rin, [], 2);
            temp = rin;
            temp(sub2ind([I I], (1: I)', rinbest)) = -Inf;
            [~, rinnextbest] = max(temp, [], 2);
            clear temp;

            if ~mod(t, 25)
                fprintf('*');
            elseif ~mod(t, 10)
                fprintf('|');
            else
                fprintf('.');
            end
        end % progress bar

        % affinity propagation done; take care of outputs
        tau = a(diag) + L(diag) - max(a(notdiag) + L(notdiag), [], 2)';
        mu = argsort(-tau);
        mu = mu(1 : K); % make exemplars the K most likely
        lik = sum(max(L(:, mu), [], 2));
        s = argmax(L(:, mu), [], 2)';
        s(mu) = 1 : length(mu); % enforce that exemplars must belong to their clusters
        return;
    end

    function affinity = patchaffinity(patch1, patch2, m)
        % function affinity = patchaffinity(patch1, patch2, m)
        %   patch1 is the candidate cluster center (k)
        %   patch2 is the training case of interest (i)

        y = 1;
        x = 2; % dimension labels
        my = m(1);
        
        if length(m) == 1
            mx = m(1);
        else
            mx = m(2);
        end
        
        m = my;

        [My1, Mx1, ~] = size(patch1);
        [My2, Mx2, ~] = size(patch2);
        M = Mx1;

        patch1 = rgb2hsv(patch1);
        patch2 = rgb2hsv(patch2);
        
        if patch1 == patch2
            affinity = NaN;
            return;
        end
        
        affinity = -inf;
        
        for x1 = 1 : (Mx1 - mx + 1)
            for x2 = floor(1 + (Mx2 - mx) / 2)
                for y1 = 1 : (My1 - my + 1)
                    for y2 = floor(1 + (My2 - my) / 2)
                        for rot = [0 90 180 270]
                            window1 = patch1(y1 : (y1 + my - 1), x1 : (x1 + mx - 1), :); % comparison data point
                            window2 = patch2(y2 : (y2 + my - 1), x2 : (x2 + mx - 1), :); % current data point

                            d = window1 - window2;
                            d(:, :, 1) = round(d(:, :, 1)) - d(:, :, 1); % wrap-around hue angles
                            d = sum(d.^2, 3); % use Gaussian model
                            d = sort(d(:));
                            d = d(1 : floor(numel(d) / 2)); % only take results better than the median
                            affinity = max(affinity, -sum(d(1 : floor(numel(d) / 2))));
                            window1 = permute(window1(end : -1 : 1, :, :), [2 1 3]); % 90-degree rotation
                        end
                    end
                end
            end
        end

        return;
    end

    % COMPUTE INDEXES FOR PATCHES (non-overlapping tiling)
    Py = [];
    Px = [];
    N = 0;

    for yi = 1 : M : height - M + 1
        for xi = 1 : M : width - M + 1
            Py = [Py yi];
            Px = [Px xi];
            N = N + 1;
        end
    end
    
    if exist(file1, 'file')
        % Load the .mat file for affinity matrix
        fprintf('The file "%s" already exist.\n', file1);
        load(file1);
        fprintf('Load the file "%s" complete.\n\n', file1);
    else
        fprintf('COMPUTING %d-by-%d AFFINITY MATRIX:  0%% done', length(Py), length(Px));
        tic;
        Lin = [];
        for i1 = 1 : N
            for i2 = [1 : i1 - 1 i1 + 1: N]
                Lin(i1, i2) = patchaffinity(img_lab(Py(i2) + (0 : M - 1), Px(i2) + (0 : M - 1), :), ...
                                            img_lab(Py(i1) + (0 : M - 1), Px(i1) + (0 : M - 1), :), m);
                if toc > 2
                    fprintf('\b\b\b\b\b\b\b\b\b%3d%% done', ...
                        floor(100 * ((i1 - 1) * N + i2) / (N * N))); 
                    tic; 
                end
            end
        end
        fprintf('\n');
        save(file1, 'Lin');
    end
    
    notdiag = [];

    for i = 1 : N
        notdiag = [notdiag; sub2ind([N N], repmat(i, [1 N - 1]), [1 : i - 1 i + 1 : N])];
    end

    
    diag = sub2ind([N N], 1 : N, 1 : N);
    Lin(diag) = median(Lin(notdiag(:))); % initialize Lii to the median of the other affinities (for lack of any better ideas)
    fprintf('Perform anffinity Propagation.\n');
    fprintf('Progress : [');
    [exemp, ~, ~] = affprop(Lin, K, 25); % perform affinity propagation
    fprintf(']\n\n');
    
    
    if exist(file2, 'file')
        % Load the .mat file for affinity matrix
        fprintf('The file "%s" already exist.\n', file2);
        load(file2);
        fprintf('Load the file "%s" complete.\n\n', file2);
    else
        fprintf('COMPUTING SEGMENTATION FOR EACH PIXEL BASED ON CLOSEST EXEMPLAR:  0%% done');
        tic;
        seg = [];
        for k = 1 : K
            for yi = 1 : height - M + 1
                for xi = 1 : width - M + 1
                    seg(yi, xi, k) = patchaffinity(img_lab(yi + (0 : M - 1), xi + (0 : M - 1 ), :), ...
                                                    img_lab(Py(exemp(k)) + (0 : M - 1), Px(exemp(k)) + (0 : M - 1), :), m);
                    if toc > 2
                        fprintf('\b\b\b\b\b\b\b\b\b%3d%% done', ...
                            floor(100 * ((k - 1) * height * width + (yi - 1) * width + (xi - 1)) / (K * height * width))); 
                        tic;
                    end

                end
            end
        end
        fprintf('\n');
        save(file2, 'seg');
    end
    [~, seg_] = max(seg, [], 3);
    fprintf('Progress the image segmentation for "%s" is complete.\n\n\n', img_file_path);
end