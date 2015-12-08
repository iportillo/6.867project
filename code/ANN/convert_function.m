function r = convert_function(func, array)
  t = num2cell(array);
  r = func(t{:});
end