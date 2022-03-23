tic
clc
clear
close all

SNR = 0;
channel_model = 'fading'; % 'awgn' or 'fading'

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

SLOT = 13; % Each frame has 20 slots
carrier.NSlot = 13;
format = 0;

%% PUCCH configuration
pucch = nrPUCCH0Config;
pucch.NSizeBWP = 273;
pucch.NStartBWP = 0;
pucch.SymbolAllocation = [12, 1]; % Time allocation
pucch.PRBSet = 0;
pucch.InitialCyclicShift = 0; % m0
pucch.FrequencyHopping = 'neither';

% Dataset Loop start here
dataset_size = 100000;
X = zeros(dataset_size, 24);
Y = zeros(dataset_size, 1);

bit_len_harq = 1;
parfor i = 1: dataset_size
    %% Transmitter
    % generate UCI bits, and the rotated sequence
    ack = randi([0, 1], bit_len_harq, 1); % [1; 0] [] for no HARQ
    sr = 1; % [] for no SR

    uci = {ack, sr}; % UCI - uplink control info
    [pucchSymbols, applied_CS] = nrPUCCH(carrier, pucch, uci);

    % Calculate index locations on the grid
    [pucchIndices,~] = nrPUCCHIndices(carrier, pucch);

    % Generate an empty grid
    pucchGrid = nrResourceGrid(carrier, nTxAnts);


    % place the UCI data in the grid
    pucchGrid(pucchIndices) = pucchSymbols;

    % Plot the grid
    %     visualize_grid(pucchGrid)

    % OFDM Modulation
    txWaveform = OFDMModulate(carrier, pucchGrid);

    % Apply the channel
    if strcmpi(channel_model, 'fading')
        rxWaveform0 = channel(txWaveform);
    else
        rxWaveform0 = txWaveform;
    end

    % Add Noise
    rxWaveform = awgn(rxWaveform0, SNR);

    %     figure()
    %     plot(abs(rxWaveform))
    %     xlabel('time'); ylabel('|rx|'); title('Rx Signals')

    % OFDM Demodulator
    rxGrid = OFDMDemodulate(carrier, rxWaveform);

    %     visualize_grid(rxGrid)

    pucchRx = nrExtractResources(pucchIndices, rxGrid);

    % Dataset X: pucchRx, Y(label): applied_CS
    % Store pucchRx and applied_CS
    X(i, :) = [real(pucchRx); imag(pucchRx)];
    Y(i) = applied_CS;
end % Dataset loop ends here
save(['./pucch_data_', num2str(SNR), 'dB_', num2str(dataset_size/1000),'k.mat'], 'X', 'Y');

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