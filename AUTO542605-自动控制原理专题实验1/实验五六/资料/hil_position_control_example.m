function hil_position_control_example
%     This example demonstrates how to do proportional control using the QUARC HIL commands.
%     It is designed to control Quanser's QUBE Servo2 USB experiment. The position of the
%     motor is read from encoder channel 0. Analog output channel 0 is used to
%     drive the motor.
% 
%     This example runs for 10 seconds. Do not press Ctrl+C.
% 
%     To generate code for this script, R2018a or later is required. For
%     the win64 target, use the following commands to build and run it:
%
%       qc_build_script('hil_position_control_example', 'win64');
%       qc_script_console('hil_position_control_example.rt-win64', 'all');
%       qc_run_script hil_position_control_example.rt-win64
%
%     If an error occurs then use quanser.hardware.hil.close_all to ensure
%     that the board is closed. A try..catch could be used instead of the err
%     output and all boards closed in the catch clause, but try..catch is not
%     compatible with MATLAB code generation.
%
%     See also:
%         QUANSER.HARDWARE.HIL.OPEN
%         QUANSER.HARDWARE.HIL.SET_ENCODER_COUNTS
%         QUANSER.HARDWARE.HIL.TASK_CREATE_ENCODER_READER
%         QUANSER.HARDWARE.HIL.WRITE_ANALOG
%         QUANSER.HARDWARE.HIL.CLOSE
%         QUANSER.HARDWARE.TASK.START
%         QUANSER.HARDWARE.TASK.READ_ENCODER
%         QUANSER.HARDWARE.TASK.STOP
%         QUANSER.HARDWARE.TASK.CLOSE
%     
%     Copyright � 2021 Quanser Inc.

board_type = 'qube_servo2_usb'; % use a QUBE-SERVO2-USB experiment
board_identifier = '0';         % use the first Q8 card in the system
clock = 0;                      % HARDWARE_CLOCK_0
frequency = 1000;               % sampling frequency
sine_frequency = 0.5;           % frequency of command signal (sine wave)
samples = -1;                   % run continuously (infinite)
samples_to_read = 1;            % number of samples read on each call to hil_task_read
encoder_channels = 0;           % encoder channel 0
analog_channels = 0;            % analog output 0
digital_channels = 0;           % digital output 0 (motor enable)
motor_enable = 1;               % value to enable motor
tf = 10;                        % final time

samples_in_buffer = 0.1 * frequency;  % number of samples in task's internal buffer (allow up to 0.1 second disruptions)
buffer_samples = 5 * frequency;       % buffer 5 seconds worth of data for plotting
    
gain = 0.3; % proportional gain
command = 0;
buffer = zeros(buffer_samples, 3); % preallocate for efficiency
index = 1;
wrap = 0;
period = 1/frequency;
   
fprintf(1, 'This example controls the QUBE Servo2 USB experiment.\n');
fprintf(1, 'Running the controller for %.3f seconds.\n', tf);
    
[board, err] = quanser.hardware.hil.open(board_type, board_identifier);
if err == 0
    time = 0;

    [task, err] = board.task_create_encoder_reader(samples_in_buffer, encoder_channels);
    if err == 0
        board.set_encoder_counts(encoder_channels, 0);
        board.write_digital(digital_channels, motor_enable);
    
        task.start(clock, frequency, samples);
        for time=0:period:tf
            % Read next sample and produce next control output
            count = task.read_encoder(samples_to_read);
        
            position  = double(count) * 360 / 2048; % convert counts to degrees
            pos_error = command - position; % compute error in position
            voltage   = gain * pos_error;   % apply proportional control
        
            err = board.write_analog(analog_channels, voltage);
    
            % Store data in a circular buffer
            buffer(index,:) = [time, command, position];
            if index == buffer_samples
                index = 1;
                wrap = 1;
            else
                index = index + 1;
            end
        
            % Compute command signal for next sampling instant
            command = 45 * sin(2*pi*sine_frequency*time);
        end
        task.stop;
        task.close;
    end

    if err < 0
        hil_print_error(err);
    end
    
    % Turn off motor
    board.write_analog(analog_channels, 0);
    board.close;

    if time > 0
        fprintf(1, 'Controller has been stopped after %.3f seconds.\n', time);
	
        % Unravel buffer
        if wrap
            buffer = [buffer(index:end,:); buffer(1:(index-1),:)];
        else
            buffer = buffer(1:index-1,:);
        end
	
        % Plot the position and command signal versus time
        if coder.target('MATLAB')
            % If running in MATLAB
            figure(1);
            plot(buffer(:, 1), buffer(:, 2:3));
            legend('Command','Measured');
        else
            % If running in real-time
            fprintf(1, '    Time  Command Position\n')
            for i=1:size(buffer, 1)
                fprintf(1, '%8.3f %8.3f %8.3f\n', buffer(i, 1), buffer(i, 2), buffer(i, 3));
            end
        end
    end
else
    hil_print_error(err);
end

