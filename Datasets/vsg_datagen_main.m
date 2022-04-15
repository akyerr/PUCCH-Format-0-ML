tic
clc
clear
close all

dataset_size = 500;

%% Carrier configuration
carrier = nrCarrierConfig;
carrier.SubcarrierSpacing = 30; % 30kHz
carrier.CyclicPrefix = "normal";
carrier.NSizeGrid = 273; % 273 resource blocks = 273*12 = 3276 resource elements (subcarriers)
carrier.NStartGrid = 0;
waveformInfo = nrOFDMInfo(carrier);

nTxAnts = 1;
nRxAnts = 1;

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
Y = zeros(dataset_size, 1);

bit_len_harq = 1;
bit_len_sr = 1;
for i = 1: dataset_size
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
    save(['./Files_for_vsg/pucch_f0_txwaveform_', num2str(i)], 'txWaveform');
    Y(i) = applied_CS;
end

save('./Files_for_vsg/pucch_f0_applied_CS', 'Y');

toc
%% Helper functions
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
