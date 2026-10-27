import React, { useRef, useState } from "react";
import { FolderUp, Link2, CheckCircle2, AlertCircle, Loader2 } from "lucide-react";
import { WorkspaceInfo } from "../types";
import { selectLocalPath, uploadFolder } from "../api";

interface FolderAttachButtonProps {
  onWorkspaceAttached: (ws: WorkspaceInfo) => void;
  attachedWorkspace: WorkspaceInfo | null;
  disabled: boolean;
}

export const FolderAttachButton: React.FC<FolderAttachButtonProps> = ({
  onWorkspaceAttached,
  attachedWorkspace,
  disabled,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [showPathInput, setShowPathInput] = useState(false);
  const [localPath, setLocalPath] = useState("examples/juice_shop");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleFolderSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsUploading(true);
    setErrorMsg(null);
    try {
      const fileList = Array.from(files);
      const ws = await uploadFolder(fileList);
      onWorkspaceAttached(ws);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to upload folder";
      setErrorMsg(msg);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleSelectPathSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!localPath.trim()) return;
    await handleQuickSelect(localPath.trim());
    setShowPathInput(false);
  };

  const handleQuickSelect = async (path: string) => {
    setIsUploading(true);
    setErrorMsg(null);
    try {
      const ws = await selectLocalPath(path);
      onWorkspaceAttached(ws);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to select path";
      setErrorMsg(msg);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFolderSelect}
          style={{ display: "none" }}
          {...({ webkitdirectory: "", directory: "", multiple: true } as unknown as Record<string, string>)}
        />

        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled || isUploading}
          className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-100 font-medium text-sm rounded-lg border border-slate-700 transition shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
          title="Attach a repository folder using directory browser"
        >
          {isUploading ? (
            <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
          ) : (
            <FolderUp className="w-4 h-4 text-blue-400" />
          )}
          <span>{isUploading ? "Uploading Folder..." : "Attach Folder"}</span>
        </button>

        <button
          type="button"
          onClick={() => setShowPathInput((prev) => !prev)}
          disabled={disabled || isUploading}
          className="flex items-center gap-1.5 px-3 py-2 bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 text-xs rounded-lg border border-slate-700 transition"
          title="Enter direct local folder path on host machine"
        >
          <Link2 className="w-3.5 h-3.5 text-slate-400" />
          <span>Local Path</span>
        </button>

        <div className="flex items-center gap-1.5 text-xs text-slate-400 bg-slate-950/40 px-2.5 py-1.5 rounded-lg border border-slate-800">
          <span className="text-[11px] text-slate-500">Quick load:</span>
          <button
            type="button"
            onClick={() => handleQuickSelect("examples/juice_shop")}
            disabled={disabled || isUploading}
            className="hover:underline text-blue-400 font-medium text-[11px] transition disabled:opacity-50"
            title="Attach examples/juice_shop scoped fixture"
          >
            examples/juice_shop
          </button>
          <span className="text-slate-600">·</span>
          <button
            type="button"
            onClick={() => handleQuickSelect("data/clones/juice-shop")}
            disabled={disabled || isUploading}
            className="hover:underline text-blue-400 font-medium text-[11px] transition disabled:opacity-50"
            title="Attach data/clones/juice-shop cloned repo"
          >
            data/clones/juice-shop
          </button>
        </div>

        {attachedWorkspace && (
          <div className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-950/60 border border-emerald-600/40 rounded-lg text-xs text-emerald-300">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
            <span className="font-semibold">{attachedWorkspace.repo_name}</span>
            <span className="text-slate-400">({attachedWorkspace.odc_report_path})</span>
          </div>
        )}
      </div>

      {!!attachedWorkspace?.upload_skipped_count && (
        <div
          role="status"
          className="flex items-start gap-2 rounded-lg border border-amber-700/60 bg-amber-950/50 p-2 text-xs text-amber-200"
        >
          <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-400" />
          <span>
            Folder upload omitted {attachedWorkspace.upload_skipped_count} files
            {attachedWorkspace.upload_skipped_files?.length
              ? `, including ${attachedWorkspace.upload_skipped_files.join(", ")}`
              : ""}
            . Use Local Path if a skipped file is needed by the run or its tests.
          </span>
        </div>
      )}

      {showPathInput && (
        <form onSubmit={handleSelectPathSubmit} className="flex items-center gap-2 mt-1">
          <input
            type="text"
            value={localPath}
            onChange={(e) => setLocalPath(e.target.value)}
            placeholder="e.g. examples/juice_shop or absolute path"
            className="flex-1 max-w-md px-3 py-1.5 bg-slate-950 border border-slate-700 rounded-lg text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
          <button
            type="submit"
            disabled={disabled || isUploading}
            className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium rounded-lg transition"
          >
            Attach Path
          </button>
        </form>
      )}

      {errorMsg && (
        <div className="flex items-center gap-2 p-2 bg-red-950/80 border border-red-700/60 rounded-lg text-xs text-red-200">
          <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}
    </div>
  );
};
