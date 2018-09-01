using System;
using System.Threading;
using UnityEngine;

public enum AsyncTaskState
{
	NotRunning, Running, Failed, Succeeded
}

public class MyAsyncTask<Tin, Tout>
{
    public bool IsCompleted { get { return _isCompleted; } }
    public AsyncTaskState State { get; internal set; }
    public Tout Result { get { return _result; } }
    public string ErrorMessage { get { return _errorMessage; } }

    public bool LogErrors { get; set; }

    private Func<Tin, Tout> _action;
    private Tout _result;
	private string _errorMessage;
    private bool _isCompleted;


    public MyAsyncTask(Func<Tin, Tout> action)
    {
        this._action = action;

        State = AsyncTaskState.NotRunning;
        _result = default(Tout);
        _isCompleted = false;

        _errorMessage = string.Empty;
        LogErrors = true;
    }

    public void Start(Tin param)
    {
        State = AsyncTaskState.Running;
        _result = default(Tout);
        _isCompleted = false;
        _errorMessage = string.Empty;

#if !NETFX_CORE
        ThreadPool.QueueUserWorkItem(state => DoInBackground(param));
#else
		System.Threading.Tasks.Task.Run(() => DoInBackground(param));
#endif	
    }

    private void DoInBackground(Tin param)
    {
        try
        {
            if (_action != null)
			{
                _result = _action(param);
			}

            State = AsyncTaskState.Succeeded;
        }
        catch (Exception ex)
        {
            State = AsyncTaskState.Failed;
			_errorMessage = ex.Message;

            if (LogErrors)
			{
                Debug.LogException(ex);
			}
        }

        _isCompleted = true;
    }

}