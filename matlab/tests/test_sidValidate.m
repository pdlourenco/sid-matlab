%% test_sidValidate - Unit tests for input parsing and validation
%
% Tests sidValidateData and sidParseOptions for correct error handling,
% default values, and name-value parsing.

fprintf('Running test_sidValidate...\n');

%% Test 1: sidValidateData — basic SISO
y = randn(1000, 1);
[y_out, ~, N, ny, nu, iTS, nT] = sidValidateData(y, []);
assert(N == 1000, 'N should be 1000');
assert(ny == 1, 'ny should be 1');
assert(nu == 0, 'nu should be 0 for time series');
assert(iTS == true, 'Empty u means time series');
assert(nT == 1, 'Single trajectory');

%% Test 2: Row vector gets converted to column
y_row = randn(1, 30);
[y_out, ~] = sidValidateData(y_row, []);
assert(isequal(size(y_out), [30, 1]), 'Row vector should be converted to column');

%% Test 3: Error on too-short data
try
    sidValidateData([1], []);
    error('Should have thrown sid:tooShort');
catch e
    assert(strcmp(e.identifier, 'sid:tooShort'), 'Expected sid:tooShort error');
end

%% Test 4: Error on complex data
try
    sidValidateData([1+1j; 2; 3], []);
    error('Should have thrown sid:complexData');
catch e
    assert(strcmp(e.identifier, 'sid:complexData'), 'Expected sid:complexData error');
end

%% Test 5: Error on NaN data
try
    sidValidateData([1; NaN; 3], []);
    error('Should have thrown sid:nonFinite');
catch e
    assert(strcmp(e.identifier, 'sid:nonFinite'), 'Expected sid:nonFinite error');
end

%% Test 6: Error on Inf data
try
    sidValidateData([1; Inf; 3], []);
    error('Should have thrown sid:nonFinite');
catch e
    assert(strcmp(e.identifier, 'sid:nonFinite'), 'Expected sid:nonFinite error');
end

%% Test 7: Error on size mismatch
try
    sidValidateData(randn(100,1), randn(50,1));
    error('Should have thrown sid:sizeMismatch');
catch e
    assert(strcmp(e.identifier, 'sid:sizeMismatch'), 'Expected sid:sizeMismatch error');
end

%% Test 8: sidValidateData multi-output
[y_out, u_out, N, ny, nu, iTS] = sidValidateData(randn(50, 2), randn(50, 3));
assert(N == 50, 'N should be 50');
assert(ny == 2, 'ny should be 2');
assert(nu == 3, 'nu should be 3');
assert(iTS == false, 'Not a time series');

%% Test 9: sidValidateData time series
[~, ~, ~, ~, nu, iTS] = sidValidateData(randn(50, 1), []);
assert(nu == 0, 'nu should be 0 for time series');
assert(iTS == true, 'Should be time series');

%% Test 10: sidParseOptions — basic usage
defs.WindowSize = 30;
defs.Frequencies = [];
defs.SampleTime = 1.0;
opts = sidParseOptions(defs, {'WindowSize', 50, 'SampleTime', 0.01});
assert(opts.WindowSize == 50, 'WindowSize should be 50');
assert(isempty(opts.Frequencies), 'Frequencies should remain []');
assert(opts.SampleTime == 0.01, 'SampleTime should be 0.01');

%% Test 11: sidParseOptions — empty args returns defaults
opts = sidParseOptions(defs, {});
assert(opts.WindowSize == 30, 'Default WindowSize should be 30');
assert(opts.SampleTime == 1.0, 'Default SampleTime should be 1.0');

%% Test 12: sidParseOptions — case insensitive
opts = sidParseOptions(defs, {'windowsize', 42});
assert(opts.WindowSize == 42, 'Case insensitive match should work');

%% Test 13: sidParseOptions — error on unknown option
try
    sidParseOptions(defs, {'BadOption', 5});
    error('Should have thrown sid:unknownOption');
catch e
    assert(strcmp(e.identifier, 'sid:unknownOption'), 'Expected sid:unknownOption error');
end

%% Test 14: sidParseOptions — error on missing value
try
    sidParseOptions(defs, {'WindowSize'});
    error('Should have thrown sid:badInput');
catch e
    assert(strcmp(e.identifier, 'sid:badInput'), 'Expected sid:badInput error');
end

%% Test 15: sidParseOptions — error on non-string key
try
    sidParseOptions(defs, {42, 'value'});
    error('Should have thrown sid:badInput');
catch e
    assert(strcmp(e.identifier, 'sid:badInput'), 'Expected sid:badInput error');
end

%% Test 16: sidFreqBT positional syntax still works
y = randn(200, 1);
u = randn(200, 1);
result = sidFreqBT(y, u, 15);
assert(result.WindowSize == 15, 'Positional M should be 15');

%% Test 17: sidFreqBT positional with frequencies
w = [0.1; 0.5; 1.0];
result = sidFreqBT(y, u, 20, w);
assert(result.WindowSize == 20, 'Positional M should be 20');
assert(isequal(result.Frequency, w), 'Positional freqs should match');

%% Test 18: sidFreqBT name-value syntax
result = sidFreqBT(y, u, 'WindowSize', 25, 'SampleTime', 0.01);
assert(result.WindowSize == 25, 'Name-value M should be 25');
assert(result.SampleTime == 0.01, 'Name-value Ts should be 0.01');

fprintf('  test_sidValidate: ALL PASSED\n');
