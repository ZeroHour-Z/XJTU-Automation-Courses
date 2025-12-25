function Quanser_PID_Control
% 直流电机位置监控系统（仿真 + 实物）界面
% 布局参照示例截图：右侧参数栏，左侧上下两块曲线+仪表盘

    %% 窗口
    fig = figure('Name', '直流电机位置监控系统', ...
        'Position', [100 100 1200 800], ...
        'NumberTitle', 'off', ...
        'Resize', 'off', ...
        'Color', [0.94 0.94 0.94]);

    %% 右侧参数设置区（侧边栏）
    panelRight = uipanel('Parent', fig, 'Position', [0.85 0.02 0.14 0.96], ...
        'BackgroundColor', 'white', 'BorderType', 'etchedin');

    % PID 参数
    hKp = uicontrol('Parent', panelRight, 'Style', 'edit', 'Position', [20 700 120 30], ...
        'String', '0.38', 'FontSize', 12);
    uicontrol('Parent', panelRight, 'Style', 'text', 'Position', [20 670 120 25], ...
        'String', 'P', 'FontSize', 10, 'BackgroundColor', 'white');

    hKi = uicontrol('Parent', panelRight, 'Style', 'edit', 'Position', [20 630 120 30], ...
        'String', '0.03', 'FontSize', 12);
    uicontrol('Parent', panelRight, 'Style', 'text', 'Position', [20 600 120 25], ...
        'String', 'I', 'FontSize', 10, 'BackgroundColor', 'white');

    hKd = uicontrol('Parent', panelRight, 'Style', 'edit', 'Position', [20 560 120 30], ...
        'String', '0.035', 'FontSize', 12);
    uicontrol('Parent', panelRight, 'Style', 'text', 'Position', [20 530 120 25], ...
        'String', 'D', 'FontSize', 10, 'BackgroundColor', 'white');

    % 仿真按钮
    uicontrol('Parent', panelRight, 'Style', 'pushbutton', 'Position', [20 180 120 40], ...
        'String', '开始 (仿真)', 'FontSize', 11, 'Callback', @startSimulation);

    % 期望角度
    hTargetAngle = uicontrol('Parent', panelRight, 'Style', 'edit', 'Position', [20 380 120 30], ...
        'String', '-180', 'FontSize', 12);
    uicontrol('Parent', panelRight, 'Style', 'text', 'Position', [10 350 140 25], ...
        'String', '期望控制角度', 'FontSize', 10, 'BackgroundColor', 'white');

    % 仿真时间
    hRunTime = uicontrol('Parent', panelRight, 'Style', 'edit', 'Position', [20 310 120 30], ...
        'String', '0.5', 'FontSize', 12);
    uicontrol('Parent', panelRight, 'Style', 'text', 'Position', [20 280 120 25], ...
        'String', '仿真时间', 'FontSize', 10, 'BackgroundColor', 'white');

    % 实物按钮
    hMotorControl = uicontrol('Parent', panelRight, 'Style', 'pushbutton', 'Position', [20 100 120 40], ...
        'String', '开始 (实物)', 'FontSize', 11, 'Callback', @motorControlCallback, 'Enable', 'off');


    %% 左侧绘图与显示区
    % 上半区：模拟
    uicontrol('Style', 'text', 'Position', [165 730 200 25], 'String', '模拟控制响应曲线', ...
        'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', [0.94 0.94 0.94]);

    axesResponse = axes('Position', [0.05 0.60 0.34 0.30]);
    grid on; hold on; box on; xlabel('时间/s'); ylabel('角度/deg');

    hSimMetrics = uicontrol('Style', 'text', 'Position', [500 520 150 150], ...
        'String', sprintf('超调量: 0.00%%\n\n峰值时间: 0.00s\n\n调节时间(2%%): 0.00s\n\n上升时间: 0.00s\n\n稳态误差: 0.00°'), ...
        'HorizontalAlignment', 'left', 'FontSize', 10, 'BackgroundColor', [0.94 0.94 0.94]);

    axSimPtr = axes('Position', [0.55 0.50 0.28 0.40]);
    axis equal; xlim([-1.2 1.2]); ylim([-1.2 1.2]); hold on; axis off;
    rectangle('Position',[-1 -1 2 2],'Curvature',[1 1],'FaceColor',[0.6 0.1 0.1],'EdgeColor','none');
    line([-1 1], [0 0], 'Color', 'w', 'LineWidth', 2);
    line([0 0], [-1 1], 'Color', 'w', 'LineWidth', 2);
    hPointer = plot(axSimPtr, [0 0], [0 0.9], 'w', 'LineWidth', 3);
    uicontrol('Style', 'text', 'Position', [770 400 100 20], 'String', '模拟运动示意图', 'FontSize', 9, 'BackgroundColor', [0.94 0.94 0.94]);
    hSimAngleDisp = uicontrol('Style', 'text', 'Position', [780 710 100 25], 'String', '0.00°', ...
        'FontSize', 12, 'ForegroundColor', 'r', 'BackgroundColor', [0.94 0.94 0.94]);

    % 下半区：实物
    uicontrol('Style', 'text', 'Position', [165 380 200 25], 'String', '实际控制响应曲线', ...
        'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', [0.94 0.94 0.94]);

    axesMotorResponse = axes('Position', [0.05 0.16 0.34 0.30]);
    grid on; hold on; box on; xlabel('时间/s'); ylabel('角度/deg');

    hMotorMetrics = uicontrol('Style', 'text', 'Position', [500 170 150 150], ...
        'String', sprintf('超调量: 0.00%%\n\n峰值时间: 0.00s\n\n调节时间(2%%): 0.00s\n\n上升时间: 0.00s\n\n稳态误差: 0.00°'), ...
        'HorizontalAlignment', 'left', 'FontSize', 10, 'BackgroundColor', [0.94 0.94 0.94]);

    axRealPtr = axes('Position', [0.55 0.06 0.28 0.40]);
    axis equal; xlim([-1.2 1.2]); ylim([-1.2 1.2]); hold on; axis off;
    rectangle('Position',[-1 -1 2 2],'Curvature',[1 1],'FaceColor',[0.6 0.1 0.1],'EdgeColor','none');
    line([-1 1], [0 0], 'Color', 'w', 'LineWidth', 2);
    line([0 0], [-1 1], 'Color', 'w', 'LineWidth', 2);
    hMotorPointer = plot(axRealPtr, [0 0], [0 0.9], 'w', 'LineWidth', 3);
    uicontrol('Style', 'text', 'Position', [770 50 100 20], 'String', '实际运动示意图', 'FontSize', 9, 'BackgroundColor', [0.94 0.94 0.94]);
    hMotorAngleDisp = uicontrol('Style', 'text', 'Position', [780 350 100 25], 'String', '0.00°', ...
        'FontSize', 12, 'ForegroundColor', 'b', 'BackgroundColor', [0.94 0.94 0.94]);

    % 句柄
    handles = guihandles(fig);
    handles.hKp = hKp; handles.hKi = hKi; handles.hKd = hKd;
    handles.hTargetAngle = hTargetAngle; handles.hRunTime = hRunTime;
    handles.axesResponse = axesResponse; handles.hPointer = hPointer; handles.hSimMetrics = hSimMetrics; handles.hSimAngleDisp = hSimAngleDisp;
    handles.hMotorControl = hMotorControl; handles.axesMotorResponse = axesMotorResponse; handles.hMotorPointer = hMotorPointer;
    handles.hMotorMetrics = hMotorMetrics; handles.hMotorAngleDisp = hMotorAngleDisp;
    handles.lastSimEnd = 0; handles.lastMotorEnd = 0;
    guidata(fig, handles);

    %% 仿真回调
    function startSimulation(hObject, ~)
        data = guidata(hObject);
        target_delta = str2double(data.hTargetAngle.String);
        t_len = str2double(data.hRunTime.String);
        start_pos = data.lastSimEnd;

        if target_delta == 0
            final_target = 0;
        else
            final_target = target_delta;
        end

        G = tf(30, [0.1 1 0]);
        C = pid(str2double(data.hKp.String), str2double(data.hKi.String), str2double(data.hKd.String));
        sys = feedback(C*G, 1);

        t_sim = 0:0.01:t_len;
        [y_raw, ~] = step(sys, t_sim);
        y_sim = start_pos + (final_target - start_pos) * y_raw;

        [ov, tr, tp, ts] = calculate_performance(t_sim, y_sim, start_pos, final_target);

        save('dynamic_response_data.mat', 't_sim', 'y_sim', 'start_pos', 'final_target', 'ov', 'tr', 'tp', 'ts', 'target_delta');
        cla(data.axesResponse);
        plot(data.axesResponse, t_sim, y_sim, 'r', 'LineWidth', 2);
        xlabel(data.axesResponse, '时间 (s)');
        ylabel(data.axesResponse, '角度 (°)');
        title(data.axesResponse, '虚拟角度响应曲线');
        grid(data.axesResponse, 'on');

        for i = 1:length(t_sim)
            theta = y_sim(i);
            set(data.hPointer, 'XData', [0 0.85*cosd(theta)], 'YData', [0 0.85*sind(theta)]);
            data.hSimAngleDisp.String = sprintf('%.2f°', theta);
            drawnow
            if mode(i,3) == 1
                drawnow;
            end
        end
        drawnow;

        data.hSimMetrics.String = sprintf(['超调量: %.2f%%\n\n' ...
            '峰值时间: %.3fs\n\n' ...
            '调节时间(2%%): %.3fs\n\n' ...
            '上升时间: %.3fs\n\n' ...
            '稳态误差: %.2f°'], ...
            ov, tp, ts, tr, abs(final_target - y_sim(end)));

        data.lastSimEnd = y_sim(end);
        guidata(hObject, data);
        data.hMotorControl.Enable = 'on';
    end

    %% 实物控制回调
    function motorControlCallback(hObject, ~)
        data = guidata(hObject);

        if ~exist('dynamic_response_data.mat', 'file')
            errordlg('请先运行虚拟仿真！');
            return;
        end

        load('dynamic_response_data.mat', 't_sim', 'y_sim', 'start_pos', 'final_target', 'ov', 'tr', 'tp', 'ts', 'target_delta');

        board_type = 'qube_servo3_usb';
        freq = 1000;
        dt = 1 / freq;

        [board, err] = quanser.hardware.hil.open(board_type, '0');
        if err ~= 0
            errordlg('硬件连接失败');
            return;
        end

        try
            board.write_digital(0, 1);
            board.write_analog(0, 0);

            [task, err] = board.task_create_encoder_reader(1000, 0);
            if err ~= 0
                error('创建编码器任务失败');
            end

            task.start(0, freq, -1);

            counts = task.read_encoder(10);
            current_motor_pos = double(mean(counts)) * 360 / 2048;

            if target_delta == 0
                motor_target = 0;
            else
                motor_target = final_target;
            end

            control_time = t_sim(end);
            actual_t = 0:dt:control_time;
            y_sim_interp = interp1(t_sim, y_sim, actual_t, 'pchip', 'extrap');

            actual_y = zeros(size(actual_t));
            previous_error = 0;
            integral_error = 0;

            Kp = str2double(data.hKp.String);
            Ki = str2double(data.hKi.String);
            Kd = str2double(data.hKd.String);

            cla(data.axesMotorResponse);
            hLine = plot(data.axesMotorResponse, 0, 0, 'b', 'LineWidth', 1.5);
            xlabel(data.axesMotorResponse, '时间 (s)');
            ylabel(data.axesMotorResponse, '角度 (°)');
            title(data.axesMotorResponse, '电机实际响应曲线');
            grid(data.axesMotorResponse, 'on');

            last_plot_time = 0;
            plot_interval = 0.05;

            for k = 1:length(actual_t)
                curr_t = actual_t(k);
                cmd = y_sim_interp(k);

                counts = task.read_encoder(3);
                pos = double(mean(counts)) * 360 / 2048;
                actual_y(k) = pos;

                error = cmd - pos;

                P_term = Kp * error;

                integral_error = integral_error + error * dt;
                max_integral = 10;
                if integral_error > max_integral
                    integral_error = max_integral;
                elseif integral_error < -max_integral
                    integral_error = -max_integral;
                end
                I_term = Ki * integral_error;

                derivative = (error - previous_error) / dt;
                alpha = 0.1;
                filtered_derivative = alpha * derivative;
                D_term = Kd * filtered_derivative;

                voltage = P_term + I_term + D_term;
                if voltage > 3
                    voltage = 3;
                elseif voltage < -3
                    voltage = -3;
                end

                board.write_analog(0, voltage);
                previous_error = error;

                if curr_t - last_plot_time >= plot_interval
                    set(hLine, 'XData', actual_t(1:k), 'YData', actual_y(1:k));
                    set(data.hMotorPointer, 'XData', [0 0.85*cosd(pos)], 'YData', [0 0.85*sind(pos)]);
                    data.hMotorAngleDisp.String = sprintf('%.2f°', pos);
                    drawnow;
                    last_plot_time = curr_t;
                end

                pause(0.001);
            end

            task.stop;
            task.close;

            if target_delta == 0
                [motor_ov, motor_tr, motor_tp, motor_ts] = calculate_performance(actual_t, actual_y, current_motor_pos, 0);
            else
                [motor_ov, motor_tr, motor_tp, motor_ts] = calculate_performance(actual_t, actual_y, current_motor_pos, motor_target);
            end

            data.hMotorMetrics.String = sprintf(['超调量: %.2f%%\n\n' ...
                '峰值时间: %.3fs\n\n' ...
                '调节时间(2%%): %.3fs\n\n' ...
                '上升时间: %.3fs\n\n' ...
                '稳态误差: %.2f°'], ...
                motor_ov, motor_tp, motor_ts, motor_tr, abs(motor_target - actual_y(end)));

            data.lastMotorEnd = actual_y(end);
            guidata(hObject, data);

        catch ME
            fprintf('控制异常: %s\n', ME.message);
            errordlg(sprintf('控制异常: %s', ME.message));
        end

        board.write_analog(0, 0);
        board.write_digital(0, 0);
        board.close;
    end

end

%% 性能指标计算函数
function [ov, tr, tp, ts] = calculate_performance(t, y, start_val, target_val)
    if abs(target_val - start_val) < 1e-6
        ov = 0; tr = 0; tp = 0; ts = 0;
        return;
    end

    y_norm = (y - start_val) / (target_val - start_val);

    if target_val > start_val
        max_val = max(y);
        ov = (max_val - target_val) / (abs(target_val)) * 100;
    else
        min_val = min(y);
        ov = (target_val - min_val) / (abs(target_val)) * 100;
    end
    if ov < 0, ov = 0; end

    idx_10 = find(y_norm >= 0.1, 1);
    idx_90 = find(y_norm >= 0.9, 1);
    if ~isempty(idx_10) && ~isempty(idx_90)
        tr = t(idx_90) - t(idx_10);
    else
        tr = 0;
    end

    if target_val > start_val
        [~, idx_max] = max(y);
    else
        [~, idx_max] = min(y);
    end
    tp = t(idx_max);

    error_band = 0.02;
    ts = 0;
    for i = 1:length(y)
        if abs(y(i) - target_val) <= abs(target_val - start_val) * error_band
            if i + 20 <= length(y)
                all_in_band = all(abs(y(i:i+20) - target_val) <= abs(target_val - start_val) * error_band);
                if all_in_band
                    ts = t(i);
                    break;
                end
            else
                ts = t(i);
                break;
            end
        end
    end
    if ts == 0
        ts = t(end);
    end
end
