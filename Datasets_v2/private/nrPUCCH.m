function varargout = nrPUCCH(carrier,pucch,uciBits,varargin)
%nrPUCCH Physical uplink control channel
%   SYM = nrPUCCH(...) returns the physical uplink control channel symbols,
%   SYM, as defined in TS 38.211 Sections 6.3.2.3 to 6.3.2.6, depending on
%   the format of physical uplink control channel. For format 0, SYM is the
%   low-PAPR sequence with sequence cyclic shift depending on the hybrid
%   repeat request acknowledgment (HARQ-ACK) bits and scheduling request
%   (SR) bit. For format 1, the HARQ-ACK or SR bits are modulated and
%   multiplied with the low-PAPR sequence and then a block-wise spreading
%   sequence is applied to return the output SYM. For format 2, the uplink
%   control information (UCI) encoded bits undergo scrambling and QPSK
%   modulation to get the output SYM. For PUCCH 3, the UCI coded bits
%   undergo scrambling, modulation, and transform precoding to get the
%   output SYM. For format 4, the UCI coded bits undergo scrambling,
%   modulation, block-wise spreading, and transform precoding to get the
%   output SYM.
%
%   SYM = nrPUCCH(CARRIER,PUCCH,UCIBITS) returns the physical uplink
%   control channel symbols, SYM, for given carrier configuration CARRIER,
%   physical uplink control channel configuration PUCCH, and UCI bits
%   UCIBITS. CARRIER is a scalar nrCarrierConfig object. For physical
%   uplink control channel formats 0, 1, 2, 3, and 4, PUCCH is a scalar
%   nrPUCCH0Config, nrPUCCH1Config, nrPUCCH2Config, nrPUCCH3Config, and
%   nrPUCCH4Config, respectively. UCIBITS is a column vector with binary
%   values or a cell array with two cells. When UCIBITS is a cell array,
%   each cell must be a column vector. For format 0, when UCIBITS is a
%   column vector or a cell array with one cell, UCIBITS is assumed to be
%   HARQ-ACK bits. For format 0, when UCIBITS is cell array with two cells,
%   the first cell is assumed as HARQ-ACK bits and the second cell is
%   assumed as SR bit. For all formats other than format 0, UCIBITS is
%   either a column vector or a cell array with one cell. For format 1,
%   UCIBITS is either HARQ-ACK or SR payload bits. In case of format 1 with
%   only positive SR, use value 0 for UCIBITS. For formats 2, 3, and 4,
%   UCIBITS is the codeword containing encoded UCI bits.
%
%   Note that for PUCCH formats 0 and 1, when GroupHopping property of
%   PUCCH configuration is set to 'disable', sequence hopping is enabled
%   which might result in selecting a sequence number that is not
%   appropriate for short base sequences.
%
%   CARRIER is a carrier configuration object, as described in <a
%   href="matlab:help('nrCarrierConfig')">nrCarrierConfig</a>.
%   Only these object properties are relevant for this function:
%
%   SubcarrierSpacing - Subcarrier spacing in kHz
%                       (15 (default), 30, 60, 120, 240)
%   CyclicPrefix      - Cyclic prefix ('normal' (default), 'extended')
%   NSizeGrid         - Number of resource blocks in carrier resource
%                       grid (1...275) (default 52)
%   NStartGrid        - Start of carrier resource grid relative to common
%                       resource block 0 (CRB 0) (0...2199) (default 0)
%   NSlot             - Slot number (default 0)
%
%   For format 0, PUCCH is the physical uplink control channel
%   configuration object, as described in <a
%   href="matlab:help('nrPUCCH0Config')">nrPUCCH0Config</a>. Only these
%   object properties are relevant for this function:
%
%   SymbolAllocation   - OFDM symbol allocation of PUCCH within a slot
%                        (default [13 1])
%   FrequencyHopping   - Frequency hopping configuration
%                        ('neither' (default), 'intraSlot', 'interSlot')
%   GroupHopping       - Group hopping configuration
%                        ('neither' (default), 'enable', 'disable')
%   HoppingID          - Hopping identity (0...1007) (default [])
%   InitialCyclicShift - Initial cyclic shift (0...11) (default 0)
%
%   For format 1, PUCCH is the physical uplink control channel
%   configuration object, as described in <a
%   href="matlab:help('nrPUCCH1Config')">nrPUCCH1Config</a>. Only these
%   object properties are relevant for this function:
%
%   SymbolAllocation   - OFDM symbol allocation of PUCCH within a slot
%                        (default [0 14])
%   FrequencyHopping   - Frequency hopping configuration
%                        ('neither' (default), 'intraSlot', 'interSlot')
%   GroupHopping       - Group hopping configuration
%                        ('neither' (default), 'enable', 'disable')
%   HoppingID          - Hopping identity (0...1007) (default [])
%   InitialCyclicShift - Initial cyclic shift (0...11) (default 0)
%   OCCI               - Orthogonal cover code index (0...6) (default 0)
%
%   For format 2, PUCCH is the physical uplink control channel
%   configuration object, as described in <a
%   href="matlab:help('nrPUCCH2Config')">nrPUCCH2Config</a>. Only these
%   object properties are relevant for this function:
%
%   NID                - Data scrambling identity (0...1023) (default [])
%   RNTI               - Radio network temporary identifier (0...65535)
%                        (default 1)
%
%   For format 3, PUCCH is the physical uplink control channel
%   configuration object, as described in <a
%   href="matlab:help('nrPUCCH3Config')">nrPUCCH3Config</a>. Only these
%   object properties are relevant for this function:
%
%   Modulation         - Modulation scheme ('QPSK' (default), 'pi/2-BPSK')
%   PRBSet             - PRBs allocated for PUCCH within the BWP
%                        (default 0)
%   NID                - Data scrambling identity (0...1023) (default [])
%   RNTI               - Radio network temporary identifier (0...65535)
%                        (default 1)
%
%   For format 4, PUCCH is the physical uplink control channel
%   configuration object, as described in <a
%   href="matlab:help('nrPUCCH4Config')">nrPUCCH4Config</a>. Only these
%   object properties are relevant for this function:
%
%   Modulation         - Modulation scheme ('QPSK' (default), 'pi/2-BPSK')
%   SpreadingFactor    - Spreading factor (2 (default), 4)
%   OCCI               - Orthogonal cover code index (0...6) (default 0)
%   NID                - Data scrambling identity (0...1023) (default [])
%   RNTI               - Radio network temporary identifier (0...65535)
%                        (default 1)
%
%   SYM = nrPUCCH(CARRIER,PUCCH,UCIBITS,NAME,VALUE) specifies an additional
%   option as a NAME,VALUE pair to allow control over the format of the
%   symbols:
%
%   'OutputDataType' - 'double' for double precision (default)
%                      'single' for single precision
%
%   % Example 1:
%   % Generate the PUCCH format 0 symbols for transmitting positive SR,
%   % in a PUCCH occupying last two OFDM symbols of a slot. The initial
%   % cyclic shift is 5, and both intra-slot frequency hopping and group
%   % hopping is enabled. Consider a carrier with 15 kHz subcarrier spacing
%   % having cell identity as 512 and slot number as 3.
%
%   % Set carrier parameters
%   carrier = nrCarrierConfig;
%   carrier.NCellID = 512;
%   carrier.SubcarrierSpacing = 15;
%   carrier.CyclicPrefix = 'normal';
%   carrier.NSlot = 3;
%
%   % Set PUCCH format 0 parameters
%   pucch0 = nrPUCCH0Config;
%   pucch0.SymbolAllocation = [12 2];
%   pucch0.FrequencyHopping = 'intraSlot';
%   pucch0.GroupHopping = 'enable';
%   pucch0.HoppingID = [];
%   pucch0.InitialCyclicShift = 5;
%
%   % Set HARQ-ACK and SR bits
%   sr = 1;
%   ack = zeros(0,1);
%   uciBits = {ack, sr};
%
%   % Get PUCCH format 0 symbols
%   sym = nrPUCCH(carrier,pucch0,uciBits);
%
%   Example 2:
%   % Generate the PUCCH format 1 modulated symbols for 1-bit UCI, when the
%   % starting symbol in a slot is 3 and the number of PUCCH symbols is 9.
%   % The orthogonal cover code index is 1, hopping identity is 512, and
%   % initial cyclic shift is 9. Consider both intra-slot frequency hopping
%   % and group hopping is enabled. Use a 60 kHz carrier with extended
%   % cyclic prefix having slot number as 7.
%
%   % Set carrier parameters
%   carrier = nrCarrierConfig;
%   carrier.SubcarrierSpacing = 60;
%   carrier.CyclicPrefix = 'extended';
%   carrier.NSlot = 7;
%
%   % Set PUCCH format 1 parameters
%   pucch1 = nrPUCCH1Config;
%   pucch1.SymbolAllocation = [3 9];
%   pucch1.FrequencyHopping = 'intraSlot';
%   pucch1.GroupHopping = 'enable';
%   pucch1.HoppingID = 512;
%   pucch1.InitialCyclicShift = 9;
%   pucch1.OCCI = 1;
%
%   % Get PUCCH format 1 symbols
%   uci = 1;
%   sym = nrPUCCH(carrier,pucch1,uci);
%
%   Example 3:
%   % Generate PUCCH format 2 symbols with cell identity as 148 and radio
%   % network temporary identifier as 160.
%
%   % Set carrier parameters
%   carrier = nrCarrierConfig;
%   carrier.NCellID = 148;
%
%   % Set PUCCH format 2 parameters
%   pucch2 = nrPUCCH2Config;
%   pucch2.NID = [];
%   pucch2.RNTI = 160;
%
%   % Get PUCCH format 2 symbols for a random codeword
%   uciCW = randi([0 1],100,1);
%   sym = nrPUCCH(carrier,pucch2,uciCW);
%
%   Example 4:
%   % Generate QPSK modulated PUCCH format 3 symbols for two resource
%   % blocks with scrambling identity as 148 and radio network temporary
%   % identifier as 1007.
%
%   % Set carrier parameters
%   carrier = nrCarrierConfig;
%   carrier.NCellID = 148;
%
%   % Set PUCCH format 3 parameters
%   pucch3 = nrPUCCH3Config;
%   pucch3.Modulation = 'QPSK';
%   pucch3.PRBSet = [0 1];
%   pucch3.NID = [];
%   pucch3.RNTI = 1007;
%
%   % Get PUCCH format 3 symbols for a random codeword
%   uciCW = randi([0 1],96,1);
%   sym = nrPUCCH(carrier,pucch3,uciCW);
%
%   Example 5:
%   % Generate pi/2-BPSK modulated PUCCH format 4 symbols with cell
%   % identity as 285, radio network temporary identifier as 897, spreading
%   % factor as 4 and orthogonal cover code sequence index as 3.
%
%   % Set carrier parameters
%   carrier = nrCarrierConfig;
%   carrier.NCellID = 285;
%
%   % Set PUCCH format 4 parameters
%   pucch4 = nrPUCCH4Config;
%   pucch4.Modulation = 'pi/2-BPSK';
%   pucch4.SpreadingFactor = 4;
%   pucch4.OCCI = 3;
%   pucch4.NID = [];
%   pucch4.RNTI = 897;
%
%   % Get PUCCH format 4 symbols for a random codeword
%   uciCW = randi([0 1],96,1);
%   sym = nrPUCCH(carrier,pucch4,uciCW);
%
%   See also nrPUCCH0, nrPUCCH1, nrPUCCH2, nrPUCCH3, nrPUCCH4, nrPUCCHDMRS,
%   nrPUCCHIndices, nrPUCCH0Config, nrPUCCH1Config, nrPUCCH2Config,
%   nrPUCCH3Config, nrPUCCH4Config.

%   Copyright 2020 The MathWorks, Inc.

%#codegen

    narginchk(3,5);

    % PV pair check
    coder.extrinsic('nr5g.internal.parseOptions');

    % Validate inputs
    fcnName = 'nrPUCCH';
    formatPUCCH = nr5g.internal.pucch.validateInputObjects(carrier,pucch);
    uciBitsCell = validateUCIBits(uciBits,formatPUCCH,fcnName);

    % Get the intra-slot frequency hopping configuration
    if strcmpi(pucch.FrequencyHopping,'intraSlot')
        intraSlotfreqHopping = 'enabled';
    else
        intraSlotfreqHopping = 'disabled';
    end

    % Get the scrambling identity
    if any(formatPUCCH == [0 1])
        if isempty(pucch.HoppingID)
            nid = double(carrier.NCellID);
        else
            nid = double(pucch.HoppingID(1));
        end
    else
        % PUCCH formats 2, 3, and 4
        if isempty(pucch.NID)
            nid = double(carrier.NCellID);
        else
            nid = double(pucch.NID(1));
        end
    end

    % Relative slot number
    nslot = mod(double(carrier.NSlot),carrier.SlotsPerFrame);

    % Get the symbols, depending on PUCCH format
    if isempty(pucch.SymbolAllocation) || (pucch.SymbolAllocation(2) == 0) ...
            || isempty(pucch.PRBSet) || isempty(uciBits)
        seq = zeros(0,1);
    else
        switch formatPUCCH
            case 0
                % PUCCH format 0
                % Get the ACK and SR bits, depending on uciBitsCell
                switch numel(uciBitsCell)
                    case 1
                        % Only one cell, treat it as ACK bits
                        ack = uciBitsCell{1};
                        sr = zeros(0,1);
                    otherwise
                        % First cell is ACK bits, second cell is SR bit
                        ack = uciBitsCell{1};
                        sr = uciBitsCell{2};
                end
                % Get the PUCCH format 0 symbols
                [seq, applied_CS, alpha_ML] = NR_PUCCH0(logical(ack(:)),logical(sr),pucch.SymbolAllocation,...
                    carrier.CyclicPrefix,nslot,nid,pucch.GroupHopping,...
                    pucch.InitialCyclicShift,intraSlotfreqHopping);
                alpha_ML = alpha_ML(pucch.SymbolAllocation(1)+1);
            case 1
                % PUCCH format 1
                % UCIBITS is either ACK or SR. Pass UCIBITS in ACK and
                % empty in SR to nrPUCCH1 function, as the function treats
                % SR as a flag and is of length 1. ACK bits in nrPUCCH1
                % function does direct processing on the bits.
                ack = uciBitsCell{:};
                sr = zeros(0,1);
                % Get the PUCCH format 1 symbols
                seq = nrPUCCH1(logical(ack(:)),sr,pucch.SymbolAllocation,...
                    carrier.CyclicPrefix,nslot,nid,pucch.GroupHopping,...
                    pucch.InitialCyclicShift,intraSlotfreqHopping,pucch.OCCI);
            case 2
                % PUCCH format 2
                % Get the PUCCH format 2 symbols
                seq = nrPUCCH2(uciBitsCell{:},nid,pucch.RNTI);
            case 3
                % PUCCH format 3
                % Get the PUCCH format 3 symbols
                seq = nrPUCCH3(uciBitsCell{:},pucch.Modulation,nid,pucch.RNTI,...
                    numel(unique(pucch.PRBSet(:))));
            otherwise
                % PUCCH format 4
                % Get the PUCCH format 4 symbols
                seq = nrPUCCH4(uciBitsCell{:},pucch.Modulation,nid,pucch.RNTI,...
                    pucch.SpreadingFactor,pucch.OCCI);
        end
    end

    % Apply options
    if nargin > 3
        opts = coder.const(nr5g.internal.parseOptions(fcnName,...
            {'OutputDataType'},varargin{:}));
        sym = cast(seq,opts.OutputDataType);
    else
        sym = seq;
    end
if formatPUCCH == 0
    varargout{1} = sym;
    varargout{2} = applied_CS;
    varargout{3} = alpha_ML;
else
    varargout{1} = sym;
end
end

function uciBitsCell = validateUCIBits(uciBits,formatPUCCH,fcnName)
%ValidateUCIBits Validates input UCIBITS and returns the UCIBITS in cell

    % Validate input UCIBITS
    validateattributes(uciBits,{'cell','numeric','logical'},{'2d'},...
        fcnName,'UCIBITS');
    maxNumElements = 1;
    if formatPUCCH == 0
        maxNumElements = 2;
    end
    if iscell(uciBits)
        numElements = numel(uciBits);
        coder.internal.errorIf(numElements > maxNumElements,...
            'nr5g:nrPUCCH:InvalidUCILength',numElements,maxNumElements);
        if numElements > 0
            validateInputWithEmpty(uciBits{1},{'double','int8','logical'},...
                {'real','column','nonnan'},fcnName,'first cell of UCIBITS');
            if any(formatPUCCH == [0 1])
                lenUCI1 = length(uciBits{1});
                coder.internal.errorIf(lenUCI1 > 2,...
                    'nr5g:nrPUCCH:InvalidUCILengthCell',lenUCI1,2);
            end
        end
        if numElements > 1
            validateInputWithEmpty(uciBits{2},{'double','int8','logical'},...
                {'real','scalar','nonnan'},fcnName,'second cell of UCIBITS');
        end
    else
        validateInputWithEmpty(uciBits,{'double','int8','logical'},...
            {'real','column','nonnan'},fcnName,'UCIBITS');
        if any(formatPUCCH == [0 1])
            lenUCI = length(uciBits);
            coder.internal.errorIf(lenUCI > 2,...
                'nr5g:nrPUCCH:InvalidUCILength',lenUCI,2);
        end
    end

    % Convert numeric type input to cell input
    if iscell(uciBits)
        uciBitsCell = uciBits;
    else
        uciBitsCell = {uciBits};
    end

end

function validateInputWithEmpty(in,classes,attributes,fcnName,varname)
%Validates input with empty handling

    if ~isempty(in)
        % Check for type and attributes
        validateattributes(in,classes,attributes,fcnName,varname);
    else
        % Check for type when input is empty
        validateattributes(in,classes,{'2d'},fcnName,varname);
    end

end
