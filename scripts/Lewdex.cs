// Lewdex.exe: a tiny launcher that opens the Lewdex window through pixi, without a console.
// install.ps1 compiles it with the C# compiler that ships with Windows (.NET Framework 4), so no
// binary lives in the repository or the release archive. It runs `pixi run --manifest-path <its folder>\pyproject.toml lewdex [args]`, so the real Python
// environment (and Triton / torch.compile) is used, at full speed.
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows.Forms;

[assembly: System.Reflection.AssemblyTitle("Lewdex")]
[assembly: System.Reflection.AssemblyProduct("Lewdex - auto-tagger for Eagle")]
[assembly: System.Reflection.AssemblyVersion("1.0.0.0")]

static class Launcher
{
    static string FindPixi()
    {
        string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        var candidates = (Environment.GetEnvironmentVariable("PATH") ?? "")
            .Split(Path.PathSeparator)
            .Where(d => d.Length > 0)
            .Select(d => Path.Combine(d.Trim('"'), "pixi.exe"))
            .Concat(new[] {
                Path.Combine(local, "pixi", "bin", "pixi.exe"),
                Path.Combine(home, ".pixi", "bin", "pixi.exe"),
            });
        foreach (string c in candidates)
        {
            try { if (File.Exists(c)) return c; } catch (ArgumentException) { }
        }
        return null;
    }

    static string Quote(string a)
    {
        return a.Length > 0 && a.IndexOfAny(new[] { ' ', '\t', '"' }) < 0
            ? a : "\"" + a.Replace("\"", "\\\"") + "\"";
    }

    [STAThread]
    static int Main(string[] args)
    {
        string dir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\');
        string manifest = Path.Combine(dir, "pyproject.toml");
        if (!File.Exists(manifest))
        {
            MessageBox.Show("pyproject.toml not found next to Lewdex.exe:\n" + dir,
                            "Lewdex", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
        if (!Directory.Exists(Path.Combine(dir, ".pixi", "envs", "default")))
        {
            MessageBox.Show("Lewdex is not installed yet.\n\nRun Install.cmd in\n" + dir,
                            "Lewdex", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return 1;
        }
        string pixi = FindPixi();
        if (pixi == null)
        {
            MessageBox.Show("pixi was not found. Run Install.cmd in\n" + dir,
                            "Lewdex", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
        string rest = string.Join(" ", args.Select(Quote));
        var psi = new ProcessStartInfo(pixi,
            "run --manifest-path " + Quote(manifest) + " lewdex " + rest)
        {
            WorkingDirectory = dir,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        try
        {
            Process.Start(psi);
        }
        catch (Exception e)
        {
            MessageBox.Show("Could not start Lewdex:\n" + e.Message, "Lewdex",
                            MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
        return 0;
    }
}
