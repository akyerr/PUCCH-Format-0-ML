tic
clc
clear
close all

SNRdB = 0:5:10;
normalize = 0; % [0, 1]
SLOT = 13;
channel_model = 'awgn'; % 'awgn' or 'fading'
num_iter = 100;


%% Carrier configuration
carrier = nrCarrierConfig;
carrier.SubcarrierSpacing = 30; % 30kHz
carrier.CyclicPrefix = "normal";
carrier.NSizeGrid = 273; % 273 resource blocks = 273*12 = 3276 resource elements (subcarriers)
carrier.NStartGrid = 0;
waveformInfo = nrOFDMInfo(carrier);

nTxAnts = 1;
nRxAnts = 1;

%% Channel Configuration
if strcmpi(channel_model, 'fading')
    % Set up TDL channel
    channel = nrTDLChannel;
    channel.DelayProfile = 'TDL-C';
    channel.DelaySpread = 300e-9;
    channel.MaximumDopplerShift = 100; % in Hz
    channel.MIMOCorrelation = 'low';
    channel.TransmissionDirection = 'Uplink';
    channel.NumTransmitAntennas = nTxAnts;
    channel.NumReceiveAntennas = nRxAnts;
    channel.SampleRate = waveformInfo.SampleRate;
    chInfo = info(channel);
    maxChDelay = ceil(max(chInfo.PathDelays*channel.SampleRate));
    maxChDelay = maxChDelay + chInfo.ChannelFilterDelay;
end


carrier.NSlot = SLOT;
format = 0;

%% PUCCH configuration
pucch = nrPUCCH0Config;
pucch.NSizeBWP = 273;
pucch.NStartBWP = 0;
pucch.SymbolAllocation = [12, 1]; % Time allocation
pucch.PRBSet = 0;
pucch.InitialCyclicShift = 0; % m0
pucch.FrequencyHopping = 'neither';

SYMB = pucch.SymbolAllocation(1);


bit_len_harq = 1;
bit_len_sr = 1;

SNR_rxwave_avg = zeros(length(SNRdB), 1);
SNR_rxgrid_avg = zeros(length(SNRdB), 1);

for s = 1:length(SNRdB)
    SNR = SNRdB(s);
    SNR_rxwave = zeros(num_iter, 1);
    SNR_rxgrid = zeros(num_iter, 1);
    for i = 1: num_iter
        %% Transmitter

        % Generate an empty grid
        pucchGrid = nrResourceGrid(carrier, nTxAnts);

        % Calculate index locations on the grid
        [pucchIndices,~] = nrPUCCHIndices(carrier, pucch);

        % generate UCI bits, and the rotated sequence
        ack = randi([0, 1], bit_len_harq, 1); % [1; 0] [] for no HARQ
        sr = randi([0, 1], bit_len_sr, 1); % [1; 0] [] for no HARQ; % [] for no SR

        uci = {ack, sr}; % UCI - uplink control info
        [pucchSymbols, applied_CS] = nrPUCCH(carrier, pucch, uci);


        % place the UCI data in the grid
        pucchGrid(pucchIndices) = pucchSymbols;


        % OFDM Modulation
        txWaveform = OFDMModulate(carrier, pucchGrid);

%         txWaveform1 = txWaveform((1:4096) + 12*(4096+288) + 352);
% 
%         tx_power = sum(abs(txWaveform1).^2)/4096;
% 
%         [txWaveform3, noise1] = AWGN(txWaveform1, SNR, 10*log10(tx_power));
% 
%         SNR3 = 10*log10((sum(abs(txWaveform1).^2)/4096)/(sum(abs(noise1).^2)/length(noise1)));
%         
% 
%         txWaveform2 = (1/sqrt(4096))*fft(txWaveform3, 4096);
%         
%         SNR4 = 10*log10((sum(abs(txWaveform2).^2)/4096)/(sum(abs(noise1).^2)/length(noise1)));

%         tx_power = sum(abs(txWaveform).^2)/61440;
%         txWaveform = txWaveform./sqrt(tx_power);

        % Apply the channel
        if strcmpi(channel_model, 'fading')
            rxWaveform0 = channel(txWaveform);
        else
            rxWaveform0 = txWaveform;
        end


        S_pucch_sig = 10*log10(sum(abs(rxWaveform0((1:4096) + 12*(4096+288) + 352)).^2)/4096);
        % Add Noise
        [rxWaveform, noise] = AWGN(rxWaveform0, SNR, S_pucch_sig);

        % For S in time domain: do we take all 61440 or just 4096 (since 1 symb
        % is occupied in this case)
        S_rxwave = sum(abs(rxWaveform0((1:4096) + 12*(4096+288) + 352)).^2)/4096;
%         S_rxwave = 10^(S_pucch_sig/10);
%         S_rxwave = 1;
        N_rxwave = sum(abs(noise((1:4096) + 12*(4096+288) + 352)).^2)/4096;

        SNR_rxwave(i) =  10*log10(S_rxwave/N_rxwave);
    

        % OFDM Demodulator
        rxGrid = OFDMDemodulate(carrier, rxWaveform);

        S_grid = sum(abs(rxGrid(1:12, 13)).^2)/12;
        N_grid = sum(sum(abs(rxGrid(13:3276, 13)).^2))/3264;

        SNR_rxgrid(i) =  10*log10(S_grid/(N_grid));
        
%         visualize_grid(rxGrid)

        pucchRx = nrExtractResources(pucchIndices, rxGrid);

        dbg = 1;
    end
    SNR_rxwave_avg(s) = mean(SNR_rxwave);
    SNR_rxgrid_avg(s) = mean(SNR_rxgrid);
    
%     figure()
%     subplot(211)
%     sgtitle(['SNR = ', num2str(SNR), ' dB'])
%     histogram(SNR_rxwave)
%     title('Rx Waveform')
%     xlabel('Measured SNR (time)')
%     ylabel('Count')
%     subplot(212)
%     histogram(SNR_rxgrid)
%     title('Rx grid')
%     xlabel('Measured SNR (freq grid)')
%     ylabel('Count')  
end

figure()
p1 = plot(SNRdB, SNR_rxgrid_avg);
p1.LineStyle = '-';
p1.Color = 'r';
p1.LineWidth = 3;
p1.Marker = 's';
p1.MarkerIndices = 1:length(SNRdB);
p1.MarkerSize = 10;
p1.MarkerFaceColor = 'r';
p1.DisplayName = 'Rx Grid SNR (Freq Domain)';

hold on

p2 = plot(SNRdB, SNR_rxwave_avg);
p2.LineStyle = '-';
p2.Color = 'b';
p2.LineWidth = 3;
p2.Marker = 'o';
p2.MarkerIndices = 1:length(SNRdB);
p2.MarkerSize = 10;
p2.MarkerFaceColor = 'b';
p2.DisplayName = 'Rx Waveform SNR (Time Domain)';


xlabel('Applied SNR (dB)')
ylabel('Measured SNR (dB)')
title('Measured SNR Averaged over multiple trials to account for fading')
grid on
lgd = legend();
lgd.FontSize = 30;
lgd.FontWeight = 'bold';

% Axis
ax = gca;
ax.XAxis.Label.FontWeight = 'bold';
ax.XAxis.LineWidth = 1.5;
ax.XAxis.FontSize = 30;
ax.XAxis.FontWeight = 'bold';

ax.YAxis.LineWidth = 1.5;
ax.YAxis.FontSize = 30;
ax.YAxis.FontWeight = 'bold';
ax.YAxis.Label.FontWeight = 'bold';

ax.Title.FontSize = 30;
ax.Title.FontWeight = 'bold';

version = 2;
h = get(0,'children');
for i=1:length(h)
    saveas(h(i), [pwd, '/Plots/figure_' num2str(i), '_v', num2str(version)], 'fig');
end

% h = get(0,'children');
% for i=1:length(h)
%     saveas(h(i), [pwd, '/Plots/figure_' num2str(i), '_v', num2str(version)], 'png');
% end

toc
%% Visualize Grid
function visualize_grid(in_grid)
figure()
imagesc(abs(in_grid(:, :, 1)));
axis xy;
xlabel('OFDM Symbols');
ylabel('Subcarriers');
xticks(1: 14)
xticklabels({'symb 0','symb 1','symb 2', 'symb 3','symb 4','symb 5', ...
    'symb 6','symb 7','symb 8', 'symb 9','symb 10','symb 11', ...
    'symb 12','symb 13'})
hold on
for i = 1:14
    line([i-0.5 i-0.5],[0 size(in_grid, 1)],'Color','white');
end
end

function tx_out = OFDMModulate(carrier, tx_grid)
SCS = carrier.SubcarrierSpacing;
Nprb = carrier.NSizeGrid;
No_RE = 12*Nprb;
NSymbPerSlot = carrier.SymbolsPerSlot;
num_ant = size(tx_grid, 3);
FFT_size = 2^ceil(log2(No_RE)); % FFT Size
SR = FFT_size*SCS; % Sample rate
slot_samples = SR*1e-3/(SCS/15e3);

CP_Len = 288*FFT_size/4096; % Cyclic prefix length % THE CP LENGTHS ARE DIFFERENT.
CP_Len1 = 352*FFT_size/4096;

ff_ind = 0:No_RE-1;
New_ff_ind = mod(ff_ind-No_RE/2, FFT_size);
RG_mat_final = zeros(FFT_size, NSymbPerSlot, num_ant);
RG_mat_final(New_ff_ind + 1 ,:,:) = tx_grid;


% Now perform IFFT and add the CP.
tx_out = zeros(slot_samples, num_ant);
start_loc = 0;
for l = 1: NSymbPerSlot
    Temp1 = sqrt(FFT_size)*ifft(RG_mat_final(:,l,:), FFT_size);

    if (l == 1) % First symbol in a slot has a different CP
        CP_len = CP_Len1;
    else
        CP_len = CP_Len;
    end
    range_loc = 1:(FFT_size+CP_len);

    tx_out(start_loc+range_loc,:) = [Temp1(FFT_size - CP_len + 1: FFT_size, 1, :); Temp1];
    start_loc = start_loc+(FFT_size+CP_len);
end
end

%% OFDM Demod
function rxGrid = OFDMDemodulate(carrier, rxWaveform)

% SCS = carrier.SubcarrierSpacing;
Nprb = carrier.NSizeGrid;
No_RE = 12*Nprb;
NSymbPerSlot = carrier.SymbolsPerSlot;
num_ant = size(rxWaveform, 2);
FFT_size = 2^ceil(log2(No_RE)); % FFT Size
% SR = FFT_size*SCS; % Sample rate
% slot_samples = SR*1e-3/(SCS/15e3);

CP_Len = 288*FFT_size/4096; % Cyclic prefix length % THE CP LENGTHS ARE DIFFERENT.
CP_Len1 = 352*FFT_size/4096;

rxGrid = zeros(No_RE, NSymbPerSlot, num_ant);

start_loc = 0;
ff_ind = 0: No_RE-1;
New_ff_ind = mod(ff_ind-No_RE/2, FFT_size);
for l = 1: NSymbPerSlot
    if(l == 1) % CP has a different length for the first symbol
        CP_len = CP_Len1;
    else
        CP_len = CP_Len;
    end
    start_loc = start_loc + CP_len;

    % CP removal and FFT
    cp_stripped = rxWaveform(start_loc + (1: FFT_size), :);
    Temp1 = sqrt(1/FFT_size)*fft(cp_stripped, FFT_size);

    % Removing the Gaurd Band and doing the inverse mapping to the RB grid.
    rxGrid(:,l,:) = Temp1(New_ff_ind + 1 , :);
    start_loc = start_loc+ FFT_size;
end
end