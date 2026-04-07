%% test_sidValidate - Unit tests for input parsing and validation
%
% Tests sidValidateData and sidParseOptions for correct error handling,
% default values, and name-value parsing.

fprintf('Running test_sidValidate...\n');
runner__nPassed = 0;

%% Test 1: sidValidateData — basic SISO
y = randn(1000, 1);
[y_out, ~, N, ny, nu, iTS, nT] = sidValidateData(y, []);
assert(N == 1000, 'N should be 1000');
assert(ny == 1, 'ny should be 1');
assert(nu == 0, 'nu should be 0 for time series');
assert(iTS == true, 'Empty u means time series');
assert(nT == 1, 'Single trajectory');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 1 passed: sidValidateData — basic SISO.\n');

%% Test 2: Row vector gets converted to column
y_row = randn(1, 30);
[y_out, ~] = sidValidateData(y_row, []);
assert(isequal(size(y_out), [30, 1]), 'Row vector should be converted to column');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 2 passed: row vector gets converted to column.\n');

%% Test 3: Error on too-short data
try
    sidValidateData([1], []);
    error('Should have thrown sid:tooShort');
catch e
    assert(strcmp(e.identifier, 'sid:tooShort'), 'Expected sid:tooShort error');
end
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 3 passed: error on too-short data.\n');

%% Test 4: Error on complex data
try
    sidValidateData([1+1j; 2; 3], []);
    error('Should have thrown sid:complexData');
catch e
    assert(strcmp(e.identifier, 'sid:complexData'), 'Expected sid:complexData error');
end
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 4 passed: error on complex data.\n');

%% Test 5: Error on NaN data
try
    sidValidateData([1; NaN; 3], []);
    error('Should have thrown sid:nonFinite');
catch e
    assert(strcmp(e.identifier, 'sid:nonFinite'), 'Expected sid:nonFinite error');
end
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 5 passed: error on NaN data.\n');

%% Test 6: Error on Inf data
try
    sidValidateData([1; Inf; 3], []);
    error('Should have thrown sid:nonFinite');
catch e
    assert(strcmp(e.identifier, 'sid:nonFinite'), 'Expected sid:nonFinite error');
end
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 6 passed: error on Inf data.\n');

%% Test 7: Error on size mismatch
try
    sidValidateData(randn(100,1), randn(50,1));
    error('Should have thrown sid:sizeMismatch');
catch e
    assert(strcmp(e.identifier, 'sid:sizeMismatch'), 'Expected sid:sizeMismatch error');
end
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 7 passed: error on size mismatch.\n');

%% Test 8: sidValidateData multi-output
[y_out, u_out, N, ny, nu, iTS] = sidValidateData(randn(50, 2), randn(50, 3));
assert(N == 50, 'N should be 50');
assert(ny == 2, 'ny should be 2');
assert(nu == 3, 'nu should be 3');
assert(iTS == false, 'Not a time series');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 8 passed: sidValidateData multi-output.\n');

%% Test 9: sidValidateData time series
[~, ~, ~, ~, nu, iTS] = sidValidateData(randn(50, 1), []);
assert(nu == 0, 'nu should be 0 for time series');
assert(iTS == true, 'Should be time series');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 9 passed: sidValidateData time series.\n');

%% Test 10: sidParseOptions — basic usage
defs.WindowSize = 30;
defs.Frequencies = [];
defs.SampleTime = 1.0;
opts = sidParseOptions(defs, {'WindowSize', 50, 'SampleTime', 0.01});
assert(opts.WindowSize == 50, 'WindowSize should be 50');
assert(isempty(opts.Frequencies), 'Frequencies should remain []');
assert(opts.SampleTime == 0.01, 'SampleTime should be 0.01');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 10 passed: sidParseOptions — basic usage.\n');

%% Test 11: sidParseOptions — empty args returns defaults
opts = sidParseOptions(defs, {});
assert(opts.WindowSize == 30, 'Default WindowSize should be 30');
assert(opts.SampleTime == 1.0, 'Default SampleTime should be 1.0');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 11 passed: sidParseOptions — empty args returns defaults.\n');

%% Test 12: sidParseOptions — case insensitive
opts = sidParseOptions(defs, {'windowsize', 42});
assert(opts.WindowSize == 42, 'Case insensitive match should work');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 12 passed: sidParseOptions — case insensitive.\n');

%% Test 13: sidParseOptions — error on unknown option
try
    sidParseOptions(defs, {'BadOption', 5});
    error('Should have thrown sid:unknownOption');
catch e
    assert(strcmp(e.identifier, 'sid:unknownOption'), 'Expected sid:unknownOption error');
end
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 13 passed: sidParseOptions — error on unknown option.\n');

%% Test 14: sidParseOptions — error on missing value
try
    sidParseOptions(defs, {'WindowSize'});
    error('Should have thrown sid:badInput');
catch e
    assert(strcmp(e.identifier, 'sid:badInput'), 'Expected sid:badInput error');
end
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 14 passed: sidParseOptions — error on missing value.\n');

%% Test 15: sidParseOptions — error on non-string key
try
    sidParseOptions(defs, {42, 'value'});
    error('Should have thrown sid:badInput');
catch e
    assert(strcmp(e.identifier, 'sid:badInput'), 'Expected sid:badInput error');
end
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 15 passed: sidParseOptions — error on non-string key.\n');

%% Test 16: sidFreqBT positional syntax still works
y = randn(200, 1);
u = randn(200, 1);
result = sidFreqBT(y, u, 15);
assert(result.WindowSize == 15, 'Positional M should be 15');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 16 passed: sidFreqBT positional syntax still works.\n');

%% Test 17: sidFreqBT positional with frequencies
w = [0.1; 0.5; 1.0];
result = sidFreqBT(y, u, 20, w);
assert(result.WindowSize == 20, 'Positional M should be 20');
assert(isequal(result.Frequency, w), 'Positional freqs should match');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 17 passed: sidFreqBT positional with frequencies.\n');

%% Test 18: sidFreqBT name-value syntax
result = sidFreqBT(y, u, 'WindowSize', 25, 'SampleTime', 0.01);
assert(result.WindowSize == 25, 'Name-value M should be 25');
assert(result.SampleTime == 0.01, 'Name-value Ts should be 0.01');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 18 passed: sidFreqBT name-value syntax.\n');

fprintf('test_sidValidate: %d/%d passed\n', runner__nPassed, runner__nPassed);
