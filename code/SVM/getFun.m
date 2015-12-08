function fun = getFun(kernel, varargin)

    if ~isempty(varargin) && ~isempty(varargin{1})
        s = varargin{1};
    else 
        s = 1;
    end
    
    if strcmp(kernel, 'linear')
        fun = @dot;
    elseif strcmp(kernel, 'gaussian')
        fun = @(x,y) exp(-(x-y)*(x-y)'/(2*s^2));
    elseif strcmp(kernel, 'poly')
        fun = @(x,y) (dot(x,y)+1)^2;
    end
end