tic
clc
clear
close all
dbstop if error
run('pucch_configs.m');

norm_tx = 0;
norm_rx = 0;

SNR = -10:5:20;
channel_model = 'fading'; % 'awgn' or 'fading'

slots = [1, 2, 3, 13, 14, 15]; % can be a vector if multiple slots are required
num_grids_per_slot = 1000;


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

rx_fail_count = zeros(length(SNR), 1);
for s = 1: length(SNR)
    SNRdB = SNR(s);
    disp(SNRdB);

    rx_fail = 0; % decoding failure counter for this SNR

    X0 = cell(num_grids_per_slot*length(slots), 1);
    Y0 = cell(num_grids_per_slot*length(slots), 1);
    parfor g = 1: num_grids_per_slot*length(slots) % this loop can be in parallel

        SLOT = slots(floor(((g-1)/num_grids_per_slot))+1);

        %% Carrier configuration
        carrier = nrCarrierConfig;
        carrier.SubcarrierSpacing = 30; % 30kHz
        carrier.CyclicPrefix = "normal";
        carrier.NSizeGrid = 273; % 273 resource blocks = 273*12 = 3276 resource elements (subcarriers)
        carrier.NStartGrid = 0;
        waveformInfo = nrOFDMInfo(carrier);
        carrier.NSlot = SLOT;

        % Generate an empty grid
        pucchGrid = nrResourceGrid(carrier, nTxAnts);

        %% Transmitter
        applied_CS0 = zeros(num_pucch_configs, 1);
        alpha_ML0 = zeros(num_pucch_configs, 1);
        for c = 1: num_pucch_configs % place all cfgs in a single grid
            pucch = pucch_cfg(c); % allocation config (location) changes each time
            % Calculate index locations on the grid
            [pucchIndices,~] = nrPUCCHIndices(carrier, pucch);

            % generate UCI bits, and the rotated sequence
            ack = randi([0, 1], bit_len_harq, 1); % [1; 0] [] for no HARQ
            sr = randi([0, 1], bit_len_sr, 1); % [1; 0] [] for no HARQ; % [] for no SR

            uci = {ack, sr}; % UCI - uplink control info
            [pucchSymbols, applied_CS0(c), alpha_ML0(c)] = nrPUCCH(carrier, pucch, uci);

            % place the UCI data in the grid
            pucchGrid(pucchIndices) = pucchSymbols;
        end


        % OFDM Modulation
        txWaveform = OFDMModulate(carrier, pucchGrid);
        
        if norm_tx == 1
            tx_power = sum(abs(txWaveform).^2)/61440;
            txWaveform = txWaveform./sqrt(tx_power);
        end


        % Apply the channel
        if strcmpi(channel_model, 'fading')
            rxWaveform0 = channel(txWaveform); %#ok
        else
            rxWaveform0 = txWaveform;
        end

        % Add Noise
        rxWaveform = awgn(rxWaveform0, SNRdB);

        %% Receiver
        % OFDM Demodulator
        rxGrid = OFDMDemodulate(carrier, rxWaveform);


        pucchRx = zeros(num_pucch_configs, 24);
        applied_CS = zeros(num_pucch_configs, 1);
        alpha_ML = zeros(num_pucch_configs, 1);
        for c = 1: num_pucch_configs % extract all cfgs in a single grid
            pucch = pucch_cfg(c);
            % Calculate index locations on the grid
            [pucchIndices,~] = nrPUCCHIndices(carrier, pucch);
            pucchRx0 = nrExtractResources(pucchIndices, rxGrid);

            % normalize to have power 1
            if norm_rx == 1
                pucch_rb_pow = sum(abs(pucchRx0).^2)/12;
                pucchRx0 = pucchRx0/sqrt(pucch_rb_pow);
            end

            pucchRx(c, :) = [real(pucchRx0); imag(pucchRx0)];
            applied_CS(c, :) = applied_CS0(c);
            alpha_ML(c, :) = alpha_ML0(c);

            recovered_CS = PUCCH0Decode(carrier, pucch, pucchRx0, SLOT);

            if recovered_CS ~= applied_CS0(c)
                rx_fail = rx_fail + 1;
            end
        end

        dbg = 1;

        X0{g} = pucchRx;
        Y0{g} = alpha_ML;
        dbg = 1;
    end
    dbg = 1;
    X1 = cell2mat(X0);
    Y1 = cell2mat(Y0);


    X = zeros(size(X1)); Y = zeros(size(Y1));

    rand_ind = randperm(size(X1, 1)).';
    X(:, :) = X1(rand_ind, :);
    Y(:, :) = Y1(rand_ind, :);

    dataset_size = size(X, 1);

    dataset_filename = ['Datafiles/Sim_data/pucch_', ...
        channel_model, '_', num2str(SNRdB), 'dB_', ...
        num2str(dataset_size/1000),'k_', ...
        'norm_tx_', num2str(norm_tx), '_', ...
        'norm_rx_', num2str(norm_rx), ...
        '.mat'];

    save(dataset_filename, 'X', 'Y');


    rx_fail_count(s) = rx_fail;
end






toc
%% Visualize Grid
function visualize_grid(in_grid)
figure()
imagesc(abs(in_grid(:, :, 1)));
axis xy;
xlabel('OFDM Symbols');
ylabel('Subcarriers');
xticks(1: 14)
% yticks(1:20:3276)
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

%% PUCCH decoding of format 0
function [recovered_CS, n_cs] = PUCCH0Decode(carrier, pucch, sym, slot)
CP_type = carrier.CyclicPrefix;
hopping_ID = carrier.NCellID;
NSymb = pucch.SymbolAllocation(2);
start_symb = pucch.SymbolAllocation(1);
group_hopping = pucch.GroupHopping;
freq_hopping = pucch.FrequencyHopping;
initial_CS = pucch.InitialCyclicShift;


load_hw_capture = 0;

if load_hw_capture == 1
    in_file = 'pucch_capture.txt';
    rx_uci_seq1 = hextxt2dec(in_file);
    rx_uci_seq2 = hextxt2dec2(in_file);
    check = sum(rx_uci_seq1 - rx_uci_seq2); %#ok
    %
    rx_uci_seq_hw = rx_uci_seq1(1: 12); % only decode 1 symbol
    rx_uci_seq = rx_uci_seq_hw;
else
    rx_uci_seq_matlab = sym(1: 12, 1); % only decode 1 symbol
    rx_uci_seq = rx_uci_seq_matlab;
end

if strcmpi(CP_type, 'normal')
    NSymbSlot = 14;
else
    NSymbSlot = 12;
end

NID = hopping_ID;
v = zeros(1, 2);
fgh = zeros(1, 2);

if strcmpi(group_hopping, 'neither') % No group hopping, no sequence hopping
    %     disp('No group hopping');
    % fgh remains [0, 0]

elseif strcmpi(group_hopping, 'enable') % Group hopping

    cinit = floor(NID/30);

    % If intra slot freq hopping is enabled, nhop = 0 for the first
    % hop and nhop = 1 for the second hop. For the first hop we need
    % c(16*slot+m) and for the second hop we need c(16*slot+8+m).
    % Where m = 0 to 7. This means we need 16 elements in total from
    % the pseudo random seq c.

    % If intra slot freq hopping is disabled, nhop is 0. But we assume
    % that hopping is enabled and generate parameters for both hops.
    % Later we decide which ones are needed.

    % First we will extract the required elements from c. In the second
    % argument [8*2*slot, 16], 8*2*slot indicates where we start extracting
    % from. 16 indicates how many elements we extract.

    c = pseudorandom_sequence(cinit, [8*2*slot, 16]);

    % reshape (c, 8, []) fixes 8 rows and then however many columns
    % that are needed to fit all the elements. We have 16 elements
    % extracted from c. The first 8 belong to the first hop. The next 8
    % belong to the second hop. The reshape results in 2 columns. The
    % first 8 elements for the first hop in col 1 and the next 8 in col
    % 2. Then weighted sum each column with powers of 2.
    fgh(1,:) = mod((2.^(0:7))*reshape(c, 8, []), 30);

else % sequence hopping

    % if group hopping is disabled, sequence hopping is enabled within
    % the group.
    cinit = 32*floor(NID/30) + mod(NID,30);

    % If intra slot freq hopping is disabled, nhop is 0. But we assume
    % that hopping is enabled and generate parameters for both hops.
    % Later we decide which ones are needed.

    % First we will extract the required elements from c. In the second
    % argument [2*slot, 2], 2*slot indicates where we start extracting
    % from. 2 indicates how many elements we extract.

    c = pseudorandom_sequence(cinit, [2*slot, 2]);

    v(1,:) = double(c');
end

% Sequence shift offset Mod 30 because there are 30 groups (0 to 29)
% This is for all cases. group hopping enable, disable and neither
fss = mod(NID,30);

% Group numbers. First element: first hop in the slot. Second
% element: second hop in the slot (0 to 29)
u = mod(fgh+fss,30);


c0 = nrPRBS(NID, [NSymbSlot*8*slot, NSymbSlot*8]);


c = pseudorandom_sequence(NID, [NSymbSlot*8*slot, NSymbSlot*8]);

check = sum(xor(c0, c));
% Get the value of ncs for all the symbols in a slot
ncs = (2.^(0:7))*reshape(c, 8, []);

% Get the cyclic shift for all the symbols in a slot
% 12 is the number of REs per resource block
m0 = initial_CS;
m11_0 = mod(m0 + ncs, 12); % for all 14 symbols

n_cs = ncs;

% extract cyclic shifts only for the PUCCH symbols in the slot
m11 = m11_0(start_symb+1: start_symb+NSymb)+1;

base_seq = low_papr_seq(u, v, 12);

base_seq_symb1 = base_seq(:, 1);


x = rx_uci_seq;
y = base_seq_symb1;

[~, ind] = max(abs(fft(x.*conj(y))));

recovered_CS = mod((ind - m11(1)), 12);

% uci = nrPUCCHDecode(carrier,pucch,[1 0],rx_uci_seq1);
dbg = 1;
end

%% Low PAPR Seq
function base_seq = low_papr_seq(u, v, m)
for vid = 1: length(v)
    for uid = 1: length(u)
        %         nIndex = (0: m-1)';
        phi = getPhiType1(u(uid), m);
        base_seq0 = exp(1j.*phi.*pi/4);
    end
end

base_seq = repmat(base_seq0, 1 , length(u));
end

%% Phi
function phi = getPhiType1(u,m)
%   PHI = getPhiType1(U,M) provides the phase values, PHI, to be applied
%   for generating the type 1 base sequence based on the group number U and
%   the sequence length M, as stated in the TS 38.211 Section 5.2.2.

% Get the table of phase values based on the length M
if m == 6 % Table 5.2.2.2-1
    phiTable = [-3  -1   3   3  -1  -3; ...
        -3   3  -1  -1   3  -3; ...
        -3  -3  -3   3   1  -3; ...
        1   1   1   3  -1  -3; ...
        1   1   1  -3  -1   3; ...
        -3   1  -1  -3  -3  -3; ...
        -3   1   3  -3  -3  -3; ...
        -3  -1   1  -3   1  -1; ...
        -3  -1  -3   1  -3  -3; ...
        -3  -3   1  -3   3  -3; ...
        -3   1   3   1  -3  -3; ...
        -3  -1  -3   1   1  -3; ...
        1   1   3  -1  -3   3; ...
        1   1   3   3  -1   3; ...
        1   1   1  -3   3  -1; ...
        1   1   1  -1   3  -3; ...
        -3  -1  -1  -1   3  -1; ...
        -3  -3  -1   1  -1  -3; ...
        -3  -3  -3   1  -3  -1; ...
        -3   1   1  -3  -1  -3; ...
        -3   3  -3   1   1  -3; ...
        -3   1  -3  -3  -3  -1; ...
        1   1  -3   3   1   3; ...
        1   1  -3  -3   1  -3; ...
        1   1   3  -1   3   3; ...
        1   1  -3   1   3   3; ...
        1   1  -1  -1   3  -1; ...
        1   1  -1   3  -1  -1; ...
        1   1  -1   3  -3  -1; ...
        1   1  -3   1  -1  -1];
elseif m == 12 % Table 5.2.2.2-2
    phiTable = [-3   1  -3  -3  -3   3  -3  -1   1   1   1  -3; ...
        -3   3   1  -3   1   3  -1  -1   1   3   3   3; ...
        -3   3   3   1  -3   3  -1   1   3  -3   3  -3; ...
        -3  -3  -1   3   3   3  -3   3  -3   1  -1  -3; ...
        -3  -1  -1   1   3   1   1  -1   1  -1  -3   1; ...
        -3  -3   3   1  -3  -3  -3  -1   3  -1   1   3; ...
        1  -1   3  -1  -1  -1  -3  -1   1   1   1  -3; ...
        -1  -3   3  -1  -3  -3  -3  -1   1  -1   1  -3; ...
        -3  -1   3   1  -3  -1  -3   3   1   3   3   1; ...
        -3  -1  -1  -3  -3  -1  -3   3   1   3  -1  -3; ...
        -3   3  -3   3   3  -3  -1  -1   3   3   1  -3; ...
        -3  -1  -3  -1  -1  -3   3   3  -1  -1   1  -3; ...
        -3  -1   3  -3  -3  -1  -3   1  -1  -3   3   3; ...
        -3   1  -1  -1   3   3  -3  -1  -1  -3  -1  -3; ...
        1   3  -3   1   3   3   3   1  -1   1  -1   3; ...
        -3   1   3  -1  -1  -3  -3  -1  -1   3   1  -3; ...
        -1  -1  -1  -1   1  -3  -1   3   3  -1  -3   1; ...
        -1   1   1  -1   1   3   3  -1  -1  -3   1  -3; ...
        -3   1   3   3  -1  -1  -3   3   3  -3   3  -3; ...
        -3  -3   3  -3  -1   3   3   3  -1  -3   1  -3; ...
        3   1   3   1   3  -3  -1   1   3   1  -1  -3; ...
        -3   3   1   3  -3   1   1   1   1   3  -3   3; ...
        -3   3   3   3  -1  -3  -3  -1  -3   1   3  -3; ...
        3  -1  -3   3  -3  -1   3   3   3  -3  -1  -3; ...
        -3  -1   1  -3   1   3   3   3  -1  -3   3   3; ...
        -3   3   1  -1   3   3  -3   1  -1   1  -1   1; ...
        -1   1   3  -3   1  -1   1  -1  -1  -3   1  -1; ...
        -3  -3   3   3   3  -3  -1   1  -3   3   1  -3; ...
        1  -1   3   1   1  -1  -1  -1   1   3  -3   1; ...
        -3   3  -3   3  -3  -3   3  -1  -1   1   3  -3];
elseif m == 18 % Table 5.2.2.2-3
    phiTable = [-1   3  -1  -3   3   1  -3  -1   3  -3  -1  -1   1   1   1  -1  -1  -1; ...
        3  -3   3  -1   1   3  -3  -1  -3  -3  -1  -3   3   1  -1   3  -3   3; ...
        -3   3   1  -1  -1   3  -3  -1   1   1   1   1   1  -1   3  -1  -3  -1; ...
        -3  -3   3   3   3   1  -3   1   3   3   1  -3  -3   3  -1  -3  -1   1; ...
        1   1  -1  -1  -3  -1   1  -3  -3  -3   1  -3  -1  -1   1  -1   3   1; ...
        3  -3   1   1   3  -1   1  -1  -1  -3   1   1  -1   3   3  -3   3  -1; ...
        -3   3  -1   1   3   1  -3  -1   1   1  -3   1   3   3  -1  -3  -3  -3; ...
        1   1  -3   3   3   1   3  -3   3  -1   1   1  -1   1  -3  -3  -1   3; ...
        -3   1  -3  -3   1  -3  -3   3   1  -3  -1  -3  -3  -3  -1   1   1   3; ...
        3  -1   3   1  -3  -3  -1   1  -3  -3   3   3   3   1   3  -3   3  -3; ...
        -3  -3  -3   1  -3   3   1   1   3  -3  -3   1   3  -1   3  -3  -3   3; ...
        -3  -3   3   3   3  -1  -1  -3  -1  -1  -1   3   1  -3  -3  -1   3  -1; ...
        -3  -1  -3  -3   1   1  -1  -3  -1  -3  -1  -1   3   3  -1   3   1   3; ...
        1   1  -3  -3  -3  -3   1   3  -3   3   3   1  -3  -1   3  -1  -3   1; ...
        -3   3  -1  -3  -1  -3   1   1  -3  -3  -1  -1   3  -3   1   3   1   1; ...
        3   1  -3   1  -3   3   3  -1  -3  -3  -1  -3  -3   3  -3  -1   1   3; ...
        -3  -1  -3  -1  -3   1   3  -3  -1   3   3   3   1  -1  -3   3  -1  -3; ...
        -3  -1   3   3  -1   3  -1  -3  -1   1  -1  -3  -1  -1  -1   3   3   1; ...
        -3   1  -3  -1  -1   3   1  -3  -3  -3  -1  -3  -3   1   1   1  -1  -1; ...
        3   3   3  -3  -1  -3  -1   3  -1   1  -1  -3   1  -3  -3  -1   3   3; ...
        -3   1   1  -3   1   1   3  -3  -1  -3  -1   3  -3   3  -1  -1  -1  -3; ...
        1  -3  -1  -3   3   3  -1  -3   1  -3  -3  -1  -3  -1   1   3   3   3; ...
        -3  -3   1  -1  -1   1   1  -3  -1   3   3   3   3  -1   3   1   3   1; ...
        3  -1  -3   1  -3  -3  -3   3   3  -1   1  -3  -1   3   1   1   3   3; ...
        3  -1  -1   1  -3  -1  -3  -1  -3  -3  -1  -3   1   1   1  -3  -3   3; ...
        -3  -3   1  -3   3   3   3  -1   3   1   1  -3  -3  -3   3  -3  -1  -1; ...
        -3  -1  -1  -3   1  -3   3  -1  -1  -3   3   3  -3  -1   3  -1  -1  -1; ...
        -3  -3   3   3  -3   1   3  -1  -3   1  -1  -3   3  -3  -1  -1  -1   3; ...
        -1  -3   1  -3  -3  -3   1   1   3   3  -3   3   3  -3  -1   3  -3   1; ...
        -3   3   1  -1  -1  -1  -1   1  -1   3   3  -3  -1   1   3  -1   3  -1];
else % m is equal to 24. Table 5.2.2.2-4
    phiTable = [-1  -3   3  -1   3   1   3  -1   1  -3  -1  -3  -1   1   3  -3  -1  -3   3   3   3  -3  -3  -3; ...
        -1  -3   3   1   1  -3   1  -3  -3   1  -3  -1  -1   3  -3   3   3   3  -3   1   3   3  -3  -3; ...
        -1  -3  -3   1  -1  -1  -3   1   3  -1  -3  -1  -1  -3   1   1   3   1  -3  -1  -1   3  -3  -3; ...
        1  -3   3  -1  -3  -1   3   3   1  -1   1   1   3  -3  -1  -3  -3  -3  -1   3  -3  -1  -3  -3; ...
        -1   3  -3  -3  -1   3  -1  -1   1   3   1   3  -1  -1  -3   1   3   1  -1  -3   1  -1  -3  -3; ...
        -3  -1   1  -3  -3   1   1  -3   3  -1  -1  -3   1   3   1  -1  -3  -1  -3   1  -3  -3  -3  -3; ...
        -3   3   1   3  -1   1  -3   1  -3   1  -1  -3  -1  -3  -3  -3  -3  -1  -1  -1   1   1  -3  -3; ...
        -3   1   3  -1   1  -1   3  -3   3  -1  -3  -1  -3   3  -1  -1  -1  -3  -1  -1  -3   3   3  -3; ...
        -3   1  -3   3  -1  -1  -1  -3   3   1  -1  -3  -1   1   3  -1   1  -1   1  -3  -3  -3  -3  -3; ...
        1   1  -1  -3  -1   1   1  -3   1  -1   1  -3   3  -3  -3   3  -1  -3   1   3  -3   1  -3  -3; ...
        -3  -3  -3  -1   3  -3   3   1   3   1  -3  -1  -1  -3   1   1   3   1  -1  -3   3   1   3  -3; ...
        -3   3  -1   3   1  -1  -1  -1   3   3   1   1   1   3   3   1  -3  -3  -1   1  -3   1   3  -3; ...
        3  -3   3  -1  -3   1   3   1  -1  -1  -3  -1   3  -3   3  -1  -1   3   3  -3  -3   3  -3  -3; ...
        -3   3  -1   3  -1   3   3   1   1  -3   1   3  -3   3  -3  -3  -1   1   3  -3  -1  -1  -3  -3; ...
        -3   1  -3  -1  -1   3   1   3  -3   1  -1   3   3  -1  -3   3  -3  -1  -1  -3  -3  -3   3  -3; ...
        -3  -1  -1  -3   1  -3  -3  -1  -1   3  -1   1  -1   3   1  -3  -1   3   1   1  -1  -1  -3  -3; ...
        -3  -3   1  -1   3   3  -3  -1   1  -1  -1   1   1  -1  -1   3  -3   1  -3   1  -1  -1  -1  -3; ...
        3  -1   3  -1   1  -3   1   1  -3  -3   3  -3  -1  -1  -1  -1  -1  -3  -3  -1   1   1  -3  -3; ...
        -3   1  -3   1  -3  -3   1  -3   1  -3  -3  -3  -3  -3   1  -3  -3   1   1  -3   1   1  -3  -3; ...
        -3  -3   3   3   1  -1  -1  -1   1  -3  -1   1  -1   3  -3  -1  -3  -1  -1   1  -3   3  -1  -3; ...
        -3  -3  -1  -1  -1  -3   1  -1  -3  -1   3  -3   1  -3   3  -3   3   3   1  -1  -1   1  -3  -3; ...
        3  -1   1  -1   3  -3   1   1   3  -1  -3   3   1  -3   3  -1  -1  -1  -1   1  -3  -3  -3  -3; ...
        -3   1  -3   3  -3   1  -3   3   1  -1  -3  -1  -3  -3  -3  -3   1   3  -1   1   3   3   3  -3; ...
        -3  -1   1  -3  -1  -1   1   1   1   3   3  -1   1  -1   1  -1  -1  -3  -3  -3   3   1  -1  -3; ...
        -3   3  -1  -3  -1  -1  -1   3  -1  -1   3  -3  -1   3  -3   3  -3  -1   3   1   1  -1  -3  -3; ...
        -3   1  -1  -3  -3  -1   1  -3  -1  -3   1   1  -1   1   1   3   3   3  -1   1  -1   1  -1  -3; ...
        -1   3  -1  -1   3   3  -1  -1  -1   3  -1  -3   1   3   1   1  -3  -3  -3  -1  -3  -1  -3  -3; ...
        3  -3  -3  -1   3   3  -3  -1   3   1   1   1   3  -1   3  -3  -1   3  -1   3   1  -1  -3  -3; ...
        -3   1  -3   1  -3   1   1   3   1  -3  -3  -1   1   3  -1  -3   3   1  -1  -3  -3  -3  -3  -3; ...
        3  -3  -1   1   3  -1  -1  -3  -1   3  -1  -3  -1  -3   3  -1   3   1   1  -3   3  -3  -3  -3];
end

% Get the phase values specific to group number u from the table of
% phase values
phi = phiTable(u+1,:)';

end
