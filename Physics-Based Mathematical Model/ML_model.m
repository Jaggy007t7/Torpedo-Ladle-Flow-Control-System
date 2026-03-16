% =========================================================================
% PHYSICS-INFORMED NEURAL NETWORK (PINN) FOR TORPEDO LADLE SYSTEM
% =========================================================================
% Architecture  : 5 -> [64, 64, 32] -> 4  (tanh activations)
% Inputs        : [t, theta, omega, p, h0]
% Outputs       : [h_hat, theta_hat, Q_hat, V_hat]
% Loss          : L_ode (ODE residuals) + L_data (vs ODE45) + L_IC (ICs)
% FPGA Export   : Fixed-point weights/biases (Q8.8 and Q16.16)
%                 + C header + VHDL inference template
% =========================================================================
% DEPENDENCIES  : No external toolboxes required (self-contained)
% =========================================================================

function normal_model()
    format long
    clc; close all;

    fprintf('=== Torpedo Ladle PINN ===\n\n');

    % ------------------------------------------------------------------
    % STEP 1 : Generate reference data using the ODE45 physics solver
    % ------------------------------------------------------------------
    fprintf('[1/5] Generating ODE45 reference data...\n');
    ref = generate_reference_data();

    % ------------------------------------------------------------------
    % STEP 2 : Build PINN and train
    % ------------------------------------------------------------------
    fprintf('[2/5] Building and training PINN...\n');
    net = train_pinn(ref);

    % ------------------------------------------------------------------
    % STEP 3 : Validate PINN against ODE45 reference
    % ------------------------------------------------------------------
    fprintf('[3/5] Validating PINN predictions...\n');
    validate_pinn(net, ref);

    % ------------------------------------------------------------------
    % STEP 4 : Export weights and biases for FPGA
    % ------------------------------------------------------------------
    fprintf('[4/5] Exporting weights/biases for FPGA...\n');
    export_fpga(net);

    % ------------------------------------------------------------------
    % STEP 5 : Generate VHDL inference template
    % ------------------------------------------------------------------
    fprintf('[5/5] Generating VHDL inference template...\n');
    generate_vhdl(net);

    fprintf('\n=== Done. Check outputs: weights.h  bias.h  nn_infer.vhd ===\n');
end


% =========================================================================
% STEP 1 HELPER : Physics ODE solver (mirrors original MATLAB model exactly)
% =========================================================================
function ref = generate_reference_data()
    c   = 0.6;
    Lf  = 20e-3;
    g   = 9.81;
    L   = 110e-3;
    R   = 60e-3;
    phi = 0;

    theta_min = 0   * pi/180;
    theta_max = 160 * pi/180;

    omega_func = @(t) 0.05 * t;

    options = odeset('Events', @(t,y) stop_event(t,y,theta_max), ...
                     'RelTol', 1e-6, 'AbsTol', 1e-8);

    y0    = [0; 0];
    tspan = 0:0.0001:50;

    [t_out, y_out] = ode45( ...
        @(t,y) system_dynamics(t,y,omega_func,c,Lf,g,L,R,phi,theta_min,theta_max), ...
        tspan, y0, options);

    theta_out = y_out(:,1);
    h_out     = y_out(:,2);
    omega_out = arrayfun(omega_func, t_out);
    omega_out(theta_out >= theta_max) = 0;

    p_out = R .* cos(theta_out + phi);
    Q_out = (2/3) .* c .* Lf .* sqrt(2 .* g .* max(0, h_out).^3);

    Vr = 2.*L.*( ...
         ((p_out+h_out)./2.*sqrt(max(0,R^2-(p_out+h_out).^2)) + (R^2/2).*asin((p_out+h_out)./R)) - ...
         (p_out./2.*sqrt(max(0,R^2-p_out.^2)) + (R^2/2).*asin(p_out./R)) );
    Vs    = L.*(p_out.*sqrt(max(0,R^2-p_out.^2)) + R^2.*asin(p_out./R) + (pi*R^2)/2);
    V_out = Vr + Vs;

    ref.t         = t_out;
    ref.theta     = theta_out;
    ref.h         = h_out;
    ref.omega     = omega_out;
    ref.p         = p_out;
    ref.Q         = Q_out;
    ref.V         = V_out;
    ref.R         = R;
    ref.L         = L;
    ref.c         = c;
    ref.Lf        = Lf;
    ref.g         = g;
    ref.phi       = phi;
    ref.theta_min = theta_min;
    ref.theta_max = theta_max;

    fprintf('   ODE45: %d time steps, final angle = %.2f deg\n', ...
        numel(t_out), theta_out(end)*180/pi);
end


% =========================================================================
% STEP 2 HELPER : Build network and train (Adam then L-BFGS)
% =========================================================================
function net = train_pinn(ref)
    n_in   = 5;
    n_out  = 4;
    hidden = [64, 64, 32];

    n_epoch_adam  = 5000;
    n_epoch_lbfgs = 1000;
    lr_adam       = 1e-3;
    lambda_ode    = 1.0;
    lambda_data   = 1.0;
    lambda_ic     = 10.0;

    % Normalisation constants stored in struct normC (avoids shadowing built-in norm)
    normC.t_max     = max(ref.t)          + 1e-10;
    normC.theta_max = ref.theta_max       + 1e-10;
    normC.omega_max = max(abs(ref.omega)) + 1e-10;
    normC.p_max     = ref.R;
    normC.h_max     = max(ref.h)          + 1e-10;
    normC.Q_max     = max(ref.Q)          + 1e-10;
    normC.V_max     = max(ref.V)          + 1e-10;

    % Xavier / Glorot initialisation
    rng(42);
    layer_sizes = [n_in, hidden, n_out];
    n_layers    = numel(layer_sizes) - 1;
    W = cell(n_layers, 1);
    b = cell(n_layers, 1);
    for k = 1:n_layers
        fan_in  = layer_sizes(k);
        fan_out = layer_sizes(k+1);
        limit   = sqrt(6 / (fan_in + fan_out));
        W{k}    = (rand(fan_out, fan_in) * 2 - 1) .* limit;
        b{k}    = zeros(fan_out, 1);
    end

    params_flat = params_to_vec(W, b);
    n_params    = numel(params_flat);
    fprintf('   PINN parameters: %d\n', n_params);

    % Training data (every 10th ODE45 point)
    idx  = 1:10:numel(ref.t);
    t_d  = ref.t(idx);
    th_d = ref.theta(idx);
    h_d  = ref.h(idx);
    om_d = ref.omega(idx);
    p_d  = ref.p(idx);
    Q_d  = ref.Q(idx);
    V_d  = ref.V(idx);
    N_d  = numel(t_d);

    % Collocation points
    t_col  = linspace(min(ref.t), max(ref.t), 2000)';
    th_col = interp1(ref.t, ref.theta, t_col);
    om_col = interp1(ref.t, ref.omega,  t_col);
    p_col  = interp1(ref.t, ref.p,      t_col);
    N_col  = numel(t_col);

    % Normalised input/output matrices
    X_data = [t_d  ./ normC.t_max,                      ...
              th_d ./ normC.theta_max,                   ...
              om_d ./ normC.omega_max,                   ...
              p_d  ./ normC.p_max,                       ...
              h_d(1) .* ones(N_d,1) ./ normC.h_max]';   % [5 x N_d]

    Y_data = [h_d  ./ normC.h_max,                      ...
              th_d ./ normC.theta_max,                   ...
              Q_d  ./ normC.Q_max,                       ...
              V_d  ./ normC.V_max]';                     % [4 x N_d]

    X_col = [t_col  ./ normC.t_max,                     ...
             th_col ./ normC.theta_max,                  ...
             om_col ./ normC.omega_max,                  ...
             p_col  ./ normC.p_max,                      ...
             h_d(1) .* ones(N_col,1) ./ normC.h_max]';  % [5 x N_col]

    % Config struct passed to all loss evaluations
    cfg.layer_sizes  = layer_sizes;
    cfg.n_layers     = n_layers;
    cfg.lambda_ode   = lambda_ode;
    cfg.lambda_data  = lambda_data;
    cfg.lambda_ic    = lambda_ic;
    cfg.normC        = normC;
    cfg.ref          = ref;
    cfg.X_data       = X_data;
    cfg.Y_data       = Y_data;
    cfg.X_col        = X_col;
    cfg.t_col        = t_col;

    % ------------------------------------------------------------------
    % Phase 1 : Adam
    % ------------------------------------------------------------------
    fprintf('   Adam phase (%d epochs)...\n', n_epoch_adam);
    m_adam       = zeros(n_params, 1);
    v_adam       = zeros(n_params, 1);
    beta1        = 0.9;
    beta2        = 0.999;
    eps_a        = 1e-8;
    loss_history = zeros(n_epoch_adam + n_epoch_lbfgs, 1);

    for epoch = 1:n_epoch_adam
        [loss, grad_val] = compute_loss_and_grad(params_flat, cfg);
        m_adam      = beta1 .* m_adam + (1 - beta1) .* grad_val;
        v_adam      = beta2 .* v_adam + (1 - beta2) .* grad_val.^2;
        m_hat       = m_adam ./ (1 - beta1^epoch);
        v_hat       = v_adam ./ (1 - beta2^epoch);
        params_flat = params_flat - lr_adam .* m_hat ./ (sqrt(v_hat) + eps_a);
        loss_history(epoch) = loss;
        if mod(epoch, 500) == 0
            fprintf('   Adam epoch %5d | Loss = %.6e\n', epoch, loss);
        end
    end

    % ------------------------------------------------------------------
    % Phase 2 : L-BFGS
    % ------------------------------------------------------------------
    fprintf('   L-BFGS phase (%d steps)...\n', n_epoch_lbfgs);
    m_mem     = 10;
    s_store   = zeros(n_params, m_mem);
    y_store   = zeros(n_params, m_mem);
    rho_store = zeros(1, m_mem);
    mem_ptr   = 0;

    for step = 1:n_epoch_lbfgs
        [loss, grad_val] = compute_loss_and_grad(params_flat, cfg);

        % Two-loop L-BFGS recursion
        q        = grad_val;
        n_stored = min(step - 1, m_mem);
        alpha_arr = zeros(n_stored, 1);
        for ii = n_stored:-1:1
            idx_m         = mod(mem_ptr - ii - 1, m_mem) + 1;
            alpha_arr(ii) = rho_store(idx_m) * (s_store(:,idx_m)' * q);
            q             = q - alpha_arr(ii) .* y_store(:,idx_m);
        end
        if mem_ptr > 0
            idx_m = mod(mem_ptr - 1, m_mem) + 1;
            gamma = (s_store(:,idx_m)' * y_store(:,idx_m)) / ...
                    (y_store(:,idx_m)' * y_store(:,idx_m) + 1e-10);
        else
            gamma = 1;
        end
        r = gamma .* q;
        for ii = 1:n_stored
            idx_m  = mod(mem_ptr - n_stored + ii - 1, m_mem) + 1;
            beta_v = rho_store(idx_m) * (y_store(:,idx_m)' * r);
            r      = r + s_store(:,idx_m) .* (alpha_arr(ii) - beta_v);
        end
        direction = -r;

        % Backtracking Armijo line search
        alpha_ls = 1.0;
        c1_ls    = 1e-4;
        for ls_iter = 1:20                          % FIX: was illegal 'for _'
            params_try = params_flat + alpha_ls .* direction;
            loss_try   = eval_loss_only(params_try, cfg);
            if loss_try <= loss + c1_ls * alpha_ls * (grad_val' * direction)
                break;
            end
            alpha_ls = alpha_ls * 0.5;
        end
        % Suppress unused-variable warning for ls_iter
        clear ls_iter;

        params_new     = params_flat + alpha_ls .* direction;
        [~, grad_new]  = compute_loss_and_grad(params_new, cfg);

        s_new     = params_new - params_flat;
        y_new     = grad_new - grad_val;
        curvature = y_new' * s_new;
        if curvature > 1e-10
            mem_ptr             = mod(mem_ptr, m_mem) + 1;
            s_store(:, mem_ptr) = s_new;
            y_store(:, mem_ptr) = y_new;
            rho_store(mem_ptr)  = 1.0 / curvature;
        end

        params_flat = params_new;
        loss_history(n_epoch_adam + step) = eval_loss_only(params_flat, cfg);

        if mod(step, 100) == 0
            fprintf('   L-BFGS step %4d | Loss = %.6e\n', step, ...
                loss_history(n_epoch_adam + step));
        end
    end

    [W_final, b_final] = vec_to_params(params_flat, layer_sizes);

    net.W            = W_final;
    net.b            = b_final;
    net.layer_sizes  = layer_sizes;
    net.n_layers     = n_layers;
    net.normC        = normC;
    net.loss_history = loss_history;

    % Plot training loss
    valid_mask = loss_history > 0;
    figure('Color', 'w');
    semilogy(find(valid_mask), loss_history(valid_mask), 'b', 'LineWidth', 1.5);
    xlabel('Epoch / Step'); ylabel('Total Loss');
    title('PINN Training Loss (Adam + L-BFGS)');
    grid on;
end


% =========================================================================
% LOSS + FINITE-DIFFERENCE GRADIENT
% =========================================================================
function [loss, grad_out] = compute_loss_and_grad(params_flat, cfg)
    loss    = eval_loss_only(params_flat, cfg);
    eps_fd  = 1e-5;
    n_p     = numel(params_flat);
    grad_out = zeros(n_p, 1);          % Pre-allocated -- avoids grow-on-loop warning
    for i = 1:n_p
        p_plus    = params_flat;  p_plus(i)  = p_plus(i)  + eps_fd;
        p_minus   = params_flat;  p_minus(i) = p_minus(i) - eps_fd;
        grad_out(i) = (eval_loss_only(p_plus,  cfg) - ...
                       eval_loss_only(p_minus, cfg)) / (2 * eps_fd);
    end
end


% =========================================================================
% SCALAR LOSS  (data + ODE residual + initial conditions)
% =========================================================================
function loss = eval_loss_only(params_flat, cfg)
    ls    = cfg.layer_sizes;
    normC = cfg.normC;
    ref   = cfg.ref;

    [W, b] = vec_to_params(params_flat, ls);

    % Data loss
    Y_pred  = nn_forward(cfg.X_data, W, b);
    L_data  = mean((Y_pred(:) - cfg.Y_data(:)).^2);

    % Initial condition loss: h(0)=0, theta(0)=0
    x_ic    = [0; 0; 0; ref.R; 0];
    y_ic    = nn_forward(x_ic, W, b);
    L_ic    = y_ic(1)^2 + y_ic(2)^2;

    % ODE residual loss (finite-difference time derivatives)
    X_col  = cfg.X_col;
    t_col  = cfg.t_col;
    N_col  = numel(t_col);
    dt     = mean(diff(t_col));

    Y_col  = nn_forward(X_col, W, b);
    h_col  = Y_col(1,:)' .* normC.h_max;
    th_col = Y_col(2,:)' .* normC.theta_max;

    dh_dt  = zeros(N_col, 1);
    dth_dt = zeros(N_col, 1);
    dh_dt(2:end-1)  = (h_col(3:end)   - h_col(1:end-2))  ./ (2*dt);
    dth_dt(2:end-1) = (th_col(3:end)  - th_col(1:end-2)) ./ (2*dt);
    dh_dt(1)   = (h_col(2)   - h_col(1))   / dt;
    dth_dt(1)  = (th_col(2)  - th_col(1))  / dt;
    dh_dt(end) = (h_col(end) - h_col(end-1)) / dt;
    dth_dt(end)= (th_col(end)- th_col(end-1))/ dt;

    omega_col = X_col(3,:)' .* normC.omega_max;
    p_col_phy = X_col(4,:)' .* normC.p_max;

    R_p   = ref.R;
    L_p   = ref.L;
    c_p   = ref.c;
    Lf_p  = ref.Lf;
    g_p   = ref.g;
    phi_p = ref.phi;

    res_theta   = dth_dt - omega_col;
    num_flow    = (2/3) .* c_p .* Lf_p .* sqrt(2 .* g_p .* max(0, h_col).^3);
    den_flow    = -2 .* L_p .* sqrt(max(0, R_p^2 - (p_col_phy + h_col).^2) + 1e-12);
    dh_physics  = num_flow ./ den_flow + R_p .* sin(th_col + phi_p) .* omega_col;
    res_h       = dh_dt - dh_physics;

    L_ode = mean(res_h.^2) + mean(res_theta.^2);

    loss = cfg.lambda_data .* L_data + ...
           cfg.lambda_ode  .* L_ode  + ...
           cfg.lambda_ic   .* L_ic;
end


% =========================================================================
% STEP 3 HELPER : Validate PINN vs ODE45
% =========================================================================
function validate_pinn(net, ref)
    normC = net.normC;
    t     = ref.t;
    N     = numel(t);
    h0_n  = ref.h(1) / normC.h_max;

    X_val = [t         ./ normC.t_max,    ...
             ref.theta ./ normC.theta_max, ...
             ref.omega ./ normC.omega_max, ...
             ref.p     ./ normC.p_max,     ...
             h0_n      .* ones(N,1)]';

    Y_pred = nn_forward(X_val, net.W, net.b);

    h_pred     = Y_pred(1,:)' .* normC.h_max;
    theta_pred = Y_pred(2,:)' .* normC.theta_max;
    Q_pred     = Y_pred(3,:)' .* normC.Q_max;
    V_pred     = Y_pred(4,:)' .* normC.V_max;

    err_h = norm_rms(h_pred - ref.h, ref.h);
    err_Q = norm_rms(Q_pred - ref.Q, ref.Q);
    err_V = norm_rms(V_pred - ref.V, ref.V);
    fprintf('   Relative RMS errors:  h=%.4f%%   Q=%.4f%%   V=%.4f%%\n', ...
        err_h*100, err_Q*100, err_V*100);

    figure('Color', 'w');
    subplot(2,2,1)
    plot(t, ref.h*1000, 'b', t, h_pred*1000, 'r--', 'LineWidth', 1.5);
    legend('ODE45','PINN'); grid on;
    title('Height h(t)'); xlabel('Time (s)'); ylabel('h (mm)');

    subplot(2,2,2)
    plot(t, ref.Q*1000, 'b', t, Q_pred*1000, 'r--', 'LineWidth', 1.5);
    legend('ODE45','PINN'); grid on;
    title('Flow Rate Q(t)'); xlabel('Time (s)'); ylabel('Q (L/s)');

    subplot(2,2,3)
    plot(t, ref.theta*180/pi, 'b', t, theta_pred*180/pi, 'r--', 'LineWidth', 1.5);
    legend('ODE45','PINN'); grid on;
    title('Angle theta(t)'); xlabel('Time (s)'); ylabel('Degrees');

    subplot(2,2,4)
    plot(t, ref.V*1000, 'b', t, V_pred*1000, 'r--', 'LineWidth', 1.5);
    legend('ODE45','PINN'); grid on;
    title('Interior Volume V(t)'); xlabel('Time (s)'); ylabel('Volume (L)');

    sgtitle('PINN vs ODE45 Validation', 'FontWeight', 'bold');
end

function e = norm_rms(diff_v, ref_v)
    e = sqrt(mean(diff_v.^2)) / (max(abs(ref_v)) + 1e-10);
end


% =========================================================================
% STEP 4 HELPER : Export weights / biases for FPGA (Q8.8 and Q16.16)
% =========================================================================
function export_fpga(net)
    W  = net.W;
    b  = net.b;
    nl = net.n_layers;
    ls = net.layer_sizes;

    scale_q88     = 256;
    int_max_q88   = 32767;
    scale_q1616   = 65536;
    int_max_q1616 = 2147483647;

    fid_w = fopen('weights.h', 'w');
    fid_b = fopen('bias.h',    'w');

    fprintf(fid_w, '/* Auto-generated by torpedo_ladle_pinn.m */\n');
    fprintf(fid_w, '/* Architecture: ');
    fprintf(fid_w, '%d ', ls);
    fprintf(fid_w, '*/\n\n');
    fprintf(fid_w, '#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n');
    fprintf(fid_w, '#include <stdint.h>\n\n');
    fprintf(fid_w, '#define N_LAYERS %d\n', nl);
    fprintf(fid_w, 'static const int LAYER_SIZES[%d] = {', nl+1);
    fprintf(fid_w, '%d,', ls(1:end-1));
    fprintf(fid_w, '%d};\n\n', ls(end));

    fprintf(fid_b, '/* Auto-generated by torpedo_ladle_pinn.m */\n');
    fprintf(fid_b, '#ifndef BIAS_H\n#define BIAS_H\n\n');
    fprintf(fid_b, '#include <stdint.h>\n\n');

    for k = 1:nl
        [rows, cols] = size(W{k});

        W_q88   = int16(max(-int_max_q88,   min(int_max_q88,   round(W{k} .* scale_q88))));
        W_q1616 = int32(max(-int_max_q1616, min(int_max_q1616, round(W{k} .* scale_q1616))));
        b_q88   = int16(max(-int_max_q88,   min(int_max_q88,   round(b{k} .* scale_q88))));
        b_q1616 = int32(max(-int_max_q1616, min(int_max_q1616, round(b{k} .* scale_q1616))));

        % Q8.8 weights
        fprintf(fid_w, '/* Layer %d  [%d x %d]  Q8.8 */\n', k, rows, cols);
        fprintf(fid_w, 'static const int16_t W%d_Q88[%d][%d] = {\n', k, rows, cols);
        for r = 1:rows
            fprintf(fid_w, '  {');
            fprintf(fid_w, '%d,', W_q88(r, 1:end-1));
            if r < rows
                fprintf(fid_w, '%d},\n', W_q88(r, end));
            else
                fprintf(fid_w, '%d}\n', W_q88(r, end));
            end
        end
        fprintf(fid_w, '};\n\n');

        % Q16.16 weights
        fprintf(fid_w, '/* Layer %d  [%d x %d]  Q16.16 */\n', k, rows, cols);
        fprintf(fid_w, 'static const int32_t W%d_Q1616[%d][%d] = {\n', k, rows, cols);
        for r = 1:rows
            fprintf(fid_w, '  {');
            fprintf(fid_w, '%d,', W_q1616(r, 1:end-1));
            if r < rows
                fprintf(fid_w, '%d},\n', W_q1616(r, end));
            else
                fprintf(fid_w, '%d}\n', W_q1616(r, end));
            end
        end
        fprintf(fid_w, '};\n\n');

        % Q8.8 biases
        fprintf(fid_b, '/* Layer %d  [%d]  Q8.8 */\n', k, rows);
        fprintf(fid_b, 'static const int16_t b%d_Q88[%d] = {', k, rows);
        fprintf(fid_b, '%d,', b_q88(1:end-1));
        fprintf(fid_b, '%d};\n\n', b_q88(end));

        % Q16.16 biases
        fprintf(fid_b, '/* Layer %d  [%d]  Q16.16 */\n', k, rows);
        fprintf(fid_b, 'static const int32_t b%d_Q1616[%d] = {', k, rows);
        fprintf(fid_b, '%d,', b_q1616(1:end-1));
        fprintf(fid_b, '%d};\n\n', b_q1616(end));

        W_recon = double(W_q88) ./ scale_q88;
        q_err   = max(abs(W{k}(:) - W_recon(:)));
        fprintf('   Layer %d  Q8.8 max quantisation error: %.6f\n', k, q_err);
    end

    normC = net.normC;
    fprintf(fid_w, '/* Input normalisation constants */\n');
    fprintf(fid_w, '#define NORM_T_MAX     %.10ef\n', normC.t_max);
    fprintf(fid_w, '#define NORM_THETA_MAX %.10ef\n', normC.theta_max);
    fprintf(fid_w, '#define NORM_OMEGA_MAX %.10ef\n', normC.omega_max);
    fprintf(fid_w, '#define NORM_P_MAX     %.10ef\n', normC.p_max);
    fprintf(fid_w, '#define NORM_H_MAX     %.10ef\n', normC.h_max);
    fprintf(fid_w, '#define NORM_Q_MAX     %.10ef\n', normC.Q_max);
    fprintf(fid_w, '#define NORM_V_MAX     %.10ef\n', normC.V_max);
    fprintf(fid_w, '\n#endif /* WEIGHTS_H */\n');
    fprintf(fid_b, '\n#endif /* BIAS_H */\n');

    fclose(fid_w);
    fclose(fid_b);
    fprintf('   Written: weights.h  bias.h\n');
end


% =========================================================================
% STEP 5 HELPER : Generate VHDL inference template
% =========================================================================
function generate_vhdl(net)
    ls = net.layer_sizes;
    nl = net.n_layers;

    fid = fopen('nn_infer.vhd', 'w');
    fprintf(fid, '-- ============================================================\n');
    fprintf(fid, '-- nn_infer.vhd  Auto-generated by torpedo_ladle_pinn.m\n');
    fprintf(fid, '-- Architecture: ');
    fprintf(fid, '%d ', ls);
    fprintf(fid, '\n');
    fprintf(fid, '-- Fixed-point: Q8.8 (16-bit signed)\n');
    fprintf(fid, '-- Activation : tanh piecewise-linear LUT\n');
    fprintf(fid, '-- ============================================================\n\n');
    fprintf(fid, 'library IEEE;\n');
    fprintf(fid, 'use IEEE.STD_LOGIC_1164.ALL;\n');
    fprintf(fid, 'use IEEE.NUMERIC_STD.ALL;\n\n');
    fprintf(fid, 'entity nn_infer is\n');
    fprintf(fid, '  Port (\n');
    fprintf(fid, '    clk       : in  std_logic;\n');
    fprintf(fid, '    rst       : in  std_logic;\n');
    fprintf(fid, '    valid_in  : in  std_logic;\n');
    fprintf(fid, '    x_t       : in  signed(15 downto 0);\n');
    fprintf(fid, '    x_theta   : in  signed(15 downto 0);\n');
    fprintf(fid, '    x_omega   : in  signed(15 downto 0);\n');
    fprintf(fid, '    x_p       : in  signed(15 downto 0);\n');
    fprintf(fid, '    x_h0      : in  signed(15 downto 0);\n');
    fprintf(fid, '    y_h       : out signed(15 downto 0);\n');
    fprintf(fid, '    y_theta   : out signed(15 downto 0);\n');
    fprintf(fid, '    y_Q       : out signed(15 downto 0);\n');
    fprintf(fid, '    y_V       : out signed(15 downto 0);\n');
    fprintf(fid, '    valid_out : out std_logic\n');
    fprintf(fid, '  );\n');
    fprintf(fid, 'end nn_infer;\n\n');
    fprintf(fid, 'architecture Behavioral of nn_infer is\n\n');
    fprintf(fid, '  constant Q_SCALE : integer := 256;\n\n');

    for k = 1:nl
        in_sz  = ls(k);
        out_sz = ls(k+1);
        fprintf(fid, '  -- Layer %d: [%d -> %d]\n', k, in_sz, out_sz);
        fprintf(fid, '  type W%d_arr is array(0 to %d, 0 to %d) of signed(15 downto 0);\n', ...
            k, out_sz-1, in_sz-1);
        fprintf(fid, '  type b%d_arr is array(0 to %d) of signed(15 downto 0);\n', k, out_sz-1);
        fprintf(fid, '  type layer%d_bus is array(0 to %d) of signed(15 downto 0);\n', k, out_sz-1);
        fprintf(fid, '  signal layer%d_out : layer%d_bus;\n\n', k, k);
    end

    fprintf(fid, '  function tanh_approx(x : signed(15 downto 0)) return signed is\n');
    fprintf(fid, '    variable xr : integer;\n');
    fprintf(fid, '  begin\n');
    fprintf(fid, '    xr := to_integer(x);\n');
    fprintf(fid, '    if    xr >  1024 then return to_signed( 256, 16);\n');
    fprintf(fid, '    elsif xr < -1024 then return to_signed(-256, 16);\n');
    fprintf(fid, '    else               return to_signed(xr * 256 / 1024, 16);\n');
    fprintf(fid, '    end if;\n');
    fprintf(fid, '  end function;\n\n');

    fprintf(fid, 'begin\n\n');
    fprintf(fid, '  process(clk)\n');
    fprintf(fid, '    variable acc : signed(47 downto 0);\n');
    fprintf(fid, '  begin\n');
    fprintf(fid, '    if rising_edge(clk) then\n');
    fprintf(fid, '      if rst = ''1'' then\n');
    fprintf(fid, '        valid_out <= ''0'';\n');
    fprintf(fid, '      elsif valid_in = ''1'' then\n');
    for k = 1:nl
        fprintf(fid, '        -- Layer %d MAC (W%d_Q88, b%d_Q88 from weights.h/bias.h)\n', k, k, k);
        fprintf(fid, '        -- for i in 0 to %d loop\n', ls(k+1)-1);
        fprintf(fid, '        --   acc := resize(b%d(i),48) * Q_SCALE;\n', k);
        fprintf(fid, '        --   for j in 0 to %d loop\n', ls(k)-1);
        fprintf(fid, '        --     acc := acc + W%d(i,j) * layer_in(j);\n', k);
        fprintf(fid, '        --   end loop;\n');
        if k < nl
            fprintf(fid, '        --   layer%d_out(i) <= tanh_approx(acc(23 downto 8));\n', k);
        else
            fprintf(fid, '        --   layer%d_out(i) <= acc(23 downto 8); -- linear\n', k);
        end
        fprintf(fid, '        -- end loop;\n\n');
    end
    fprintf(fid, '        valid_out <= ''1'';\n');
    fprintf(fid, '      end if;\n');
    fprintf(fid, '    end if;\n');
    fprintf(fid, '  end process;\n\n');
    fprintf(fid, '  y_h     <= layer%d_out(0);\n', nl);
    fprintf(fid, '  y_theta <= layer%d_out(1);\n', nl);
    fprintf(fid, '  y_Q     <= layer%d_out(2);\n', nl);
    fprintf(fid, '  y_V     <= layer%d_out(3);\n', nl);
    fprintf(fid, '\nend Behavioral;\n');
    fclose(fid);
    fprintf('   Written: nn_infer.vhd\n');
end


% =========================================================================
% NEURAL NETWORK FORWARD PASS
% =========================================================================
function Y = nn_forward(X, W, b)
    % X: [n_in x N],  Y: [n_out x N]
    A        = X;
    n_layers = numel(W);
    for k = 1:n_layers - 1
        Z = W{k} * A + b{k};
        A = tanh(Z);
    end
    Y = W{end} * A + b{end};   % Linear output layer
end


% =========================================================================
% PARAMETER PACKING  (weights + biases <--> flat vector)
% =========================================================================
function v = params_to_vec(W, b)
    % Pre-allocate full vector to avoid size-change-on-loop warning
    n_layers = numel(W);
    total    = 0;
    for k = 1:n_layers
        total = total + numel(W{k}) + numel(b{k});
    end
    v   = zeros(total, 1);
    ptr = 1;
    for k = 1:n_layers
        nw = numel(W{k});
        nb = numel(b{k});
        v(ptr : ptr+nw-1) = W{k}(:);   ptr = ptr + nw;
        v(ptr : ptr+nb-1) = b{k}(:);   ptr = ptr + nb;
    end
end

function [W, b] = vec_to_params(v, layer_sizes)
    n_layers = numel(layer_sizes) - 1;
    W   = cell(n_layers, 1);
    b   = cell(n_layers, 1);
    ptr = 1;
    for k = 1:n_layers
        fan_out = layer_sizes(k+1);
        fan_in  = layer_sizes(k);
        n_w     = fan_out * fan_in;
        W{k}    = reshape(v(ptr : ptr+n_w-1), fan_out, fan_in);
        ptr     = ptr + n_w;
        b{k}    = v(ptr : ptr+fan_out-1);
        ptr     = ptr + fan_out;
    end
end


% =========================================================================
% ODE PHYSICS  (identical to original solve_ode_system.m)
% =========================================================================
function dydt = system_dynamics(t, y, omega_func, c, Lf, g, L, R, phi, theta_min, theta_max) 
    theta = y(1);
    h     = y(2);
    if theta >= theta_max
        omega         = 0;
        current_theta = theta_max;
    else
        omega         = omega_func(t);
        current_theta = theta;
    end
    dtheta_dt = omega;
    if current_theta < theta_min
        dh_dt = 0;
    else
        p            = R * cos(current_theta + phi);
        num_f        = (2/3) * c * Lf * sqrt(2 * g * max(0, h)^3);
        inner_sq     = max(0, R^2 - (p + h)^2);
        den_f        = -2 * L * sqrt(inner_sq + 1e-12);
        outflow_term = num_f / den_f;
        kin_term     = R * sin(current_theta + phi) * omega;
        dh_dt        = outflow_term + kin_term;
    end
    dydt = [dtheta_dt; dh_dt];
end

function [value, isterminal, direction] = stop_event(~, y, theta_max)
    % First argument (t) intentionally unused -- replaced with ~
    theta = y(1);
    h     = y(2);
    if theta >= (theta_max - 1e-5)
        value = h - 1e-5;
    else
        value = 1;
    end
    isterminal = 1;
    direction  = -1;
end
