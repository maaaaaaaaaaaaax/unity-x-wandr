using UnityEngine;
using System.Collections.Concurrent;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public class TCPManager : MonoBehaviour
{
    private TcpClient client;
    private NetworkStream stream;
    private Thread receiveThread;
    private volatile bool running = true;

    // The receive thread and Unity's main thread run in parallel.
    // Complete JSON frames are stored in a thread-safe queue instead of keeping
    // only "the last string", which could accidentally replay old frames.
    private readonly ConcurrentQueue<string> receivedFrames = new ConcurrentQueue<string>();

    public static TCPManager Instance;

    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
            return;
        }
    }

    void Start()
    {
        try
        {
            client = new TcpClient("127.0.0.1", 5005);
            stream = client.GetStream();

            receiveThread = new Thread(ReceiveData);
            receiveThread.IsBackground = true;
            receiveThread.Start();
        }
        catch (System.Exception e)
        {
            Debug.LogError("TCP Error: " + e.Message);
        }
    }

    public void SendMessageToPython(string message)
    {
        if (client == null || stream == null || message == null) return;

        if (!message.EndsWith("\n"))
        {
            message += "\n";
        }

        byte[] data = Encoding.UTF8.GetBytes(message);
        stream.Write(data, 0, data.Length);
        Debug.Log("Sent to Python: " + message.TrimEnd('\n', '\r'));
    }

    private void ReceiveData()
    {
        // TCP does not preserve message boundaries: one read can contain half a
        // JSON object, exactly one JSON object, or multiple JSON objects. Python
        // terminates every frame with '\n', so bytes are buffered until the next
        // complete line.
        byte[] buffer = new byte[8192];
        StringBuilder dataBuffer = new StringBuilder();

        while (running)
        {
            if (stream == null)
            {
                Thread.Sleep(1);
                continue;
            }

            try
            {
                int bytesRead = stream.Read(buffer, 0, buffer.Length);
                if (bytesRead <= 0) continue;

                dataBuffer.Append(Encoding.UTF8.GetString(buffer, 0, bytesRead));

                string[] messages = dataBuffer.ToString().Split('\n');
                for (int i = 0; i < messages.Length - 1; i++)
                {
                    if (!string.IsNullOrWhiteSpace(messages[i]))
                    {
                        // Every complete line is one motion frame. Enqueuing
                        // instead of overwriting prevents frame drops when one
                        // TCP packet contains multiple Python frames.
                        receivedFrames.Enqueue(messages[i]);
                    }
                }

                // The final split segment is either empty or an incomplete JSON
                // frame, so keep it buffered for the next read.
                dataBuffer.Clear();
                dataBuffer.Append(messages[messages.Length - 1]);
            }
            catch (System.Exception e)
            {
                if (running)
                {
                    Debug.LogError("TCP Receive Error: " + e.Message);
                }
            }
        }
    }

    public bool TryGetNextFrame(out string json)
    {
        // Unity calls this from Update() and consumes at most one new frame per
        // Unity frame. If the queue is empty, SMPLXController keeps the current
        // target pose.
        return receivedFrames.TryDequeue(out json);
    }

    public string GetReceivedData()
    {
        // Legacy wrapper for code that still calls GetReceivedData().
        // New call sites should use TryGetNextFrame() so it is clear that the
        // returned frame is removed from the queue.
        if (TryGetNextFrame(out string json))
        {
            return json;
        }

        return "";
    }

    void OnDestroy()
    {
        running = false;

        try
        {
            stream?.Close();
            client?.Close();
        }
        catch
        {
            // Ignore shutdown races.
        }

        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Join(200);
        }
    }
}
