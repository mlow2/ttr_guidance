import itertools
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpatheffects

# 12 maximally separated hues, tuned for deuteranopia/protanopia.
COLOR_CODES = [
    "#00cc4a",
    "#4c4cff",
    "#18d5f3",
    "#FFCA1A",
    "#B84DFF",
    "#E673D9",
    "#E67317",
    "#A1E617",
    "#17E6A1",
    "#178CE6",
    "#5C17E6",
    "#E617BA",
]



def create_trajectory_figure():
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlim((-8, 6))
    ax.set_ylim((-6, 6))
    ax.set_aspect("equal")
    return fig, ax


def plot_traj(ax, xs_sim, vehicle_index, safety_violation=None, zorder=4,
              no_fade_safety_violation=False, priority_rank_1based=None):
    """Fading scatter trajectory; the palette entry follows priority rank when given."""
    _n = xs_sim.shape[1]

    alpha_start = 0.00
    alpha_end = 0.25
    alphas = np.linspace(alpha_start, alpha_end, _n)

    if priority_rank_1based is not None:
        ci = (int(priority_rank_1based) - 1) % len(COLOR_CODES)
    else:
        ci = vehicle_index % len(COLOR_CODES)
    color = COLOR_CODES[ci]

    if safety_violation is None:
        violated = np.zeros(_n, dtype=bool)
    else:
        violated = np.asarray(safety_violation, dtype=bool)
    if no_fade_safety_violation:
        alphas = np.where(violated, alpha_end, alphas)

    # Violations are drawn in a second pass so red is always visible.
    for is_violation_pass in (False, True):
        m = violated == is_violation_pass
        if not np.any(m):
            continue
        ax.scatter(
            xs_sim[0, m],
            xs_sim[1, m],
            color='red' if is_violation_pass else color,
            s=20,
            alpha=alphas[m],
            linewidths=0,
            zorder=zorder + 1 if is_violation_pass else zorder,
        )
    ax.scatter(
        xs_sim[0, 0],
        xs_sim[1, 0],
        color='red' if violated[0] else color,
        s=30,
        alpha=1.0,
        linewidths=1,
        edgecolor='k',
        zorder=zorder + 1 if violated[0] else zorder,
    )


def plot_control_history(
    num_vehicles,
    ts_sim,
    us_sim,
    u_ref_list=None,
    apply_smoothing=False,
    convolution_window=10):
    vehicle_u_dim = us_sim.shape[0] // num_vehicles

    if u_ref_list is not None:
        u_ref = np.stack(u_ref_list, axis=1)
    else:
        u_ref = None

    fig, axs = plt.subplots(us_sim.shape[0], 1, figsize=(12, 2.5 * us_sim.shape[0]), sharex=True)
    if us_sim.shape[0] == 1:
        axs = [axs]

    for i in range(num_vehicles):
        us_sim_i = us_sim[vehicle_u_dim*i:vehicle_u_dim*(i+1), :]
        u_ref_i = u_ref[vehicle_u_dim*i:vehicle_u_dim*(i+1), :] if u_ref is not None else None
        for j in range(vehicle_u_dim):
            us_sim_ij = us_sim_i[j, :]
            u_ref_ij = u_ref_i[j, :] if u_ref is not None else None
            if apply_smoothing:
                us_sim_ij = np.convolve(us_sim_ij, np.ones(convolution_window)/convolution_window, mode='same')
                if u_ref is not None:
                    u_ref_ij = np.convolve(u_ref_ij, np.ones(convolution_window)/convolution_window, mode='same')
            axs[vehicle_u_dim*i+j].plot(ts_sim, us_sim_ij, label=f'Vehicle {i} control')
            if u_ref is not None:
                axs[vehicle_u_dim*i+j].plot(ts_sim, u_ref_ij, '--', label=f'Vehicle {i} u_ref')
            axs[vehicle_u_dim*i+j].set_ylabel(f'u[{i}][{j}]')
            axs[vehicle_u_dim*i+j].grid(True)
    axs[-1].set_xlabel('Time')

    fig.suptitle('Control input (us_sim) and reference (u_ref) for each vehicle')
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    return fig


def plot_ttr_history(
    ts_sim,
    ttr_array,
    priority_list,
    ttr_target_array=None
):
    num_vehicles = ttr_array.shape[1]
    fig, ax = plt.subplots(num_vehicles+2, 1, figsize=(10, 4 * (num_vehicles + 2)))
    ax[0].plot(ts_sim, ttr_array)
    ax[0].legend()
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('TTR')
    ax[0].grid(True)
    for i in range(len(priority_list)-1):
        ttr_diff = ttr_array[:,priority_list[i+1]] - ttr_array[:,priority_list[i]]
        ax[1].plot(ts_sim, ttr_diff, label=f'TTR diff {i+1}')
    ax[1].legend()
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('TTR diff')
    ax[1].grid(True)
    for i in range(num_vehicles):
        ax[i+2].plot(ts_sim, ttr_array[:, i], 'k', label=f'TTR {i}')
        if ttr_target_array is not None:
            ax[i+2].plot(ts_sim, ttr_target_array[:, i], 'k:', label=f'TTR target {i}')
        ax[i+2].legend()
        ax[i+2].set_xlabel('Time')
        ax[i+2].set_ylabel('TTR target')
        ax[i+2].grid(True)
    return fig

def draw_corridors(ax, corridors, label=True, color="0.35"):
    for k, c in enumerate(corridors):
        c.visualize_corridor(ax)
        if label:
            center = np.asarray(c.entrance_position, dtype=float)
            dir_perp = np.array([-np.sin(c.heading), np.cos(c.heading)])
            lab = center + (c.half_width + 0.35) * dir_perp
            ax.text(lab[0], lab[1], str(k), color=color, fontsize=9,
                    fontweight="bold", ha="center", va="center", zorder=0.6)
    return ax


def build_single_corridor_title(*, scenario_name, nominal_mode, safety_mode,
                                ttr_separation, seed=None, num_vehicles=None):
    """Two-line title: mode axes on line 1, run axes on line 2."""
    nominal = nominal_mode
    line1 = f"Single corridor | {safety_mode} | {nominal}"
    parts = [rf"$\Delta_t$ = {ttr_separation:g} s", str(scenario_name)]
    if seed is not None:
        parts.append(f"seed {seed}")
    if num_vehicles:
        parts.append(f"{num_vehicles} vehicles")
    return f"{line1}\n{' | '.join(parts)}"


def animate_vehicle_trajectories(
    vehicle_x_dim: int,
    xs_sim,
    num_vehicles,
    output_path,
    *,
    separation_distance,
    title,
    verbosity,
    x_lim=(-8, 5),
    y_lim=(-6, 6),
    fps=10,
    frame_stride=1,
    trace_alpha=0.4,
    trace_size=12,
    vehicle_size=120,
    priority_list=None,
    ts_sim=None,
    corridor_handle=None,
    mission_complete_list=None,
    mission_complete_alpha=0.2,
    critical_neighbor_per_step=None,
    critical_h_per_step=None,
    safety_override_per_step=None,
    ttr_target_capped_per_step=None,
    ttr_per_step=None,
    ttr_target_per_step=None,
    fs_priority_per_step=None,
    status_per_step=None,
):
    """Animate multi-vehicle trajectories to an mp4: fading trails plus a heading-aligned body.

    Args:
        xs_sim: (state_dim, num_steps), state_dim = vehicle_x_dim * num_vehicles.
        separation_distance: pairwise separation radius drawn around each body;
            pairs closer than it are flagged. Pairs with a vehicle inside the
            corridor or completed are not checked.
        title: title text; the frame time is appended.
        verbosity: 0 draws the vehicle id on the body; 1 adds, rotated with
            heading above it, the priority rank plus live TTR; 2 adds a
            right-side per-vehicle metrics panel.
        priority_list: per-step priority orderings; the rank labels each body.
        frame_stride: take every Nth simulation frame.
        corridor_handle: CorridorHandle drawn as a static background.
        critical_neighbor_per_step, critical_h_per_step, safety_override_per_step:
            per-step per-vehicle safety diagnostics drawn as a directed line from
            each ego to its most critical neighbour (crimson while the safety
            filter overrides the ego, grey otherwise; grey only at verbosity 2).
        ttr_target_capped_per_step: highlights the panel ``tgt`` cell where the
            TTR target is capped or the TTR is non-finite.
        ttr_per_step, ttr_target_per_step, fs_priority_per_step, status_per_step:
            per-step per-vehicle series shown in the verbosity labels and panel.
    """
    if verbosity not in (0, 1, 2):
        raise ValueError(f"verbosity must be 0-2, got {verbosity}")
    state_dim, num_steps = xs_sim.shape
    assert state_dim == vehicle_x_dim * num_vehicles

    handles = [corridor_handle] if corridor_handle is not None else []

    if frame_stride > 1:
        frame_indices = list(range(0, num_steps, frame_stride))
        xs_sim = xs_sim[:, frame_indices]
        ts_sim = [ts_sim[i] for i in frame_indices] if ts_sim is not None else None
        priority_list = [priority_list[i] for i in frame_indices] if priority_list is not None else None
        mission_complete_list = [mission_complete_list[i] for i in frame_indices] if mission_complete_list is not None else None
        ttr_per_step = [ttr_per_step[i] for i in frame_indices] if ttr_per_step is not None else None
        ttr_target_per_step = [ttr_target_per_step[i] for i in frame_indices] if ttr_target_per_step is not None else None
        fs_priority_per_step = [fs_priority_per_step[i] for i in frame_indices] if fs_priority_per_step is not None else None
        status_per_step = [status_per_step[i] for i in frame_indices] if status_per_step is not None else None
        critical_neighbor_per_step = [critical_neighbor_per_step[i] for i in frame_indices] if critical_neighbor_per_step is not None else None
        critical_h_per_step = [critical_h_per_step[i] for i in frame_indices] if critical_h_per_step is not None else None
        safety_override_per_step = [safety_override_per_step[i] for i in frame_indices] if safety_override_per_step is not None else None
        ttr_target_capped_per_step = [ttr_target_capped_per_step[i] for i in frame_indices] if ttr_target_capped_per_step is not None else None
        num_steps = len(frame_indices)

    if priority_list is not None and priority_list[0] is not None:
        init_list = priority_list[0]
        rank_of_vehicle = [init_list.index(i) + 1 for i in range(num_vehicles)]
    else:
        rank_of_vehicle = [i + 1 for i in range(num_vehicles)]

    # Color follows the displayed priority rank, not the vehicle id.
    vehicle_colors = [
        COLOR_CODES[(rank_of_vehicle[i] - 1) % len(COLOR_CODES)]
        for i in range(num_vehicles)
    ]

    base_title = title
    n_title_lines = base_title.count("\n") + 1

    _panel = verbosity == 2
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]
    max_dim = 10.0
    aspect = x_range / y_range
    if aspect >= 1.0:
        fig_w = max_dim
        fig_h = max_dim / aspect
    else:
        fig_h = max_dim
        fig_w = max_dim * aspect
    # The panel takes a fixed right margin; widen the figure so the axes keep their size.
    right = 0.62 if _panel else 0.97
    if _panel:
        fig_w *= 1.9
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    top = 0.92 - 0.04 * (n_title_lines - 1)
    fig.subplots_adjust(left=0.08, right=right, bottom=0.06, top=top)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal")

    title_fontsize = plt.rcParams['axes.titlesize']
    if n_title_lines > 1:
        # Shrink multi-line titles to the axes width so the longest line does not overrun the panel.
        import matplotlib.font_manager as _fm
        base_pt = _fm.FontProperties(
            size=plt.rcParams['axes.titlesize']).get_size_in_points()
        ax_w_in = ax.get_position().width * fig.get_size_inches()[0]
        lines = base_title.split("\n")
        longest = max(len(lines[0]), len(lines[-1]) + len(" - Time: 000.00"))
        fit_pt = 0.94 * ax_w_in * 72.0 / (0.6 * max(longest, 1))
        title_fontsize = max(7.0, min(base_pt, fit_pt))
    title_text = ax.text(
        0.5, 1.02, base_title,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=title_fontsize,
        fontweight=plt.rcParams['axes.titleweight'])
    ax.set_title("")

    def _set_ticks_and_grid_include_origin():
        ax.set_axisbelow(True)

        x_ticks = np.asarray(ax.get_xticks(), dtype=float)
        y_ticks = np.asarray(ax.get_yticks(), dtype=float)

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        if x0 <= 0.0 <= x1 and not np.any(np.isclose(x_ticks, 0.0)):
            x_ticks = np.append(x_ticks, 0.0)
        if y0 <= 0.0 <= y1 and not np.any(np.isclose(y_ticks, 0.0)):
            y_ticks = np.append(y_ticks, 0.0)

        # set_xticks widens the view to include out-of-range ticks; keep the requested limits.
        x_ticks = x_ticks[(x_ticks >= x0) & (x_ticks <= x1)]
        y_ticks = y_ticks[(y_ticks >= y0) & (y_ticks <= y1)]
        ax.set_xticks(np.sort(x_ticks))
        ax.set_yticks(np.sort(y_ticks))

        ax.grid(True, which="major", color="0.88", linewidth=0.8, zorder=0)

    _set_ticks_and_grid_include_origin()

    if handles:
        draw_corridors(ax, handles)

    def _completed_at(k, i):
        return (mission_complete_list is not None and k < len(mission_complete_list)
                and mission_complete_list[k] is not None and mission_complete_list[k][i])

    group_per_step = []
    for k in range(num_steps):
        g = [0] * num_vehicles
        for i in range(num_vehicles):
            if _completed_at(k, i):
                g[i] = -1
            elif handles:
                state_i = xs_sim[vehicle_x_dim * i: vehicle_x_dim * (i + 1), k]
                if any(h.check_in_corridor(state_i) for h in handles):
                    g[i] = -1
        group_per_step.append(g)

    _violation = np.zeros((num_vehicles, num_steps), dtype=bool)
    for k in range(num_steps):
        g = group_per_step[k]
        positions_k = np.array([
            xs_sim[vehicle_x_dim * i : vehicle_x_dim * i + 2, k]
            for i in range(num_vehicles)
        ])
        for i, j in itertools.combinations(range(num_vehicles), 2):
            if g[i] < 0 or g[i] != g[j]:
                continue
            if np.linalg.norm(positions_k[i] - positions_k[j]) < separation_distance:
                _violation[i, k] = True
                _violation[j, k] = True

    import matplotlib.colors as mcolors
    _vehicle_rgba = np.array([mcolors.to_rgba(c) for c in vehicle_colors])
    _red_rgba = np.array(mcolors.to_rgba("red"))

    # Two trail layers per vehicle so violation dots render above every normal trail.
    trace_scatters = []
    trace_viol_scatters = []
    for i in range(num_vehicles):
        sc = ax.scatter([], [], s=trace_size, zorder=1)
        trace_scatters.append(sc)
        sc_v = ax.scatter([], [], s=trace_size, zorder=2)
        trace_viol_scatters.append(sc_v)

    _half_len = 0.5
    _half_width = 0.25
    _rect_verts = [
        (_half_len, _half_width), (-_half_len, _half_width),
        (-_half_len, -_half_width), (_half_len, -_half_width), (_half_len, _half_width),
    ]
    _rect_codes = [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.LINETO, mpath.Path.LINETO, mpath.Path.CLOSEPOLY]
    _rect_path = mpath.Path(_rect_verts, _rect_codes)

    current_scatters = []
    for i in range(num_vehicles):
        sc = ax.scatter([], [], s=vehicle_size, c=[vehicle_colors[i]], edgecolors="k", linewidths=1, zorder=3)
        current_scatters.append(sc)

    id_texts = []
    for i in range(num_vehicles):
        t = ax.text(0, 0, "", fontsize=6.5, fontweight="bold", color="white",
                    ha="center", va="center", rotation_mode="anchor", zorder=7)
        t.set_path_effects([mpatheffects.withStroke(linewidth=1.4, foreground="black")])
        id_texts.append(t)

    separation_patches = []
    for _ in range(num_vehicles):
        circle = mpatches.Circle(
            (0.0, 0.0),
            radius=separation_distance,
            edgecolor="k",
            facecolor="none",
            linewidth=1.0,
            alpha=0.7,
            zorder=2,
        )
        ax.add_patch(circle)
        separation_patches.append(circle)

    # The perpendicular offset keeps a mutually-critical pair readable as two opposed arrows.
    _CRIT_GREY = "0.35"
    _CRIT_RED = "crimson"
    _CRIT_PERP_FRAC = 0.04
    critical_lines = []
    critical_marks = []
    for i in range(num_vehicles):
        ln, = ax.plot([], [], lw=0.8, color=_CRIT_GREY, alpha=0.85, zorder=2.5)
        mk, = ax.plot([], [], ls="none", color=_CRIT_GREY, ms=6, zorder=2.6)
        critical_lines.append(ln)
        critical_marks.append(mk)

    def _render_critical_links(frame, positions, mc_row):
        show_all = verbosity == 2
        for i in range(num_vehicles):
            ln, mk = critical_lines[i], critical_marks[i]
            j = _series_val(critical_neighbor_per_step, frame, i)
            overriding = bool(_series_val(safety_override_per_step, frame, i))
            hidden = (
                j is None
                or int(j) >= num_vehicles
                or (not show_all and not overriding)
                or _completed_at(frame, i)
                or (mc_row is not None and mc_row[i]))
            if hidden:
                ln.set_data([], [])
                mk.set_data([], [])
                continue
            p, q = positions[i], positions[int(j)]
            dx, dy = q[0] - p[0], q[1] - p[1]
            seg = float(np.hypot(dx, dy))
            if seg <= 0.0:
                ln.set_data([], [])
                mk.set_data([], [])
                continue
            off = _CRIT_PERP_FRAC * seg
            nx, ny = -dy / seg * off, dx / seg * off
            ln.set_data([p[0] + nx, q[0] + nx], [p[1] + ny, q[1] + ny])
            mk.set_data([0.5 * (p[0] + q[0]) + nx], [0.5 * (p[1] + q[1]) + ny])
            mk.set_marker((3, 0, np.degrees(np.arctan2(dy, dx)) - 90.0))
            color = _CRIT_RED if overriding else _CRIT_GREY
            ln.set_color(color)
            mk.set_color(color)

    # Offset by a fraction of the separation distance so the label stays inside the circle.
    _rich_body_fs = 7
    _label_perp_off = 0.40 * separation_distance

    def _series_val(per_step, frame, i):
        if per_step is None or frame >= len(per_step) or per_step[frame] is None:
            return None
        row = per_step[frame]
        return row[i] if i < len(row) else None

    rich_texts = []
    for i in range(num_vehicles):
        t = ax.text(0, 0, "", fontsize=_rich_body_fs, fontweight="bold",
                    color="black", ha="center", va="bottom",
                    rotation_mode="anchor", zorder=6)
        rich_texts.append(t)

    _panel_fs = 11
    _panel_text = None
    _panel_capped_text = None
    _panel_header = (f"{'id':>2}  {'C.P':<5} {'st':<4} {'TTR':>5} {'tgt':>5} "
                     f"{'dTTR':>5} {'nn':>6} {'crit':>10}")
    if _panel:
        _panel_text = fig.text(
            0.625, 0.85, "", ha="left", va="top",
            fontfamily="monospace", fontsize=_panel_fs, zorder=5)
        # Same anchor and monospace font, so the capped `tgt` cells overlay the base table exactly.
        _panel_capped_text = fig.text(
            0.625, 0.85, "", ha="left", va="top", color="darkorange",
            fontfamily="monospace", fontsize=_panel_fs, zorder=6)

    def _fmt_num(v, width, sign=False):
        if v is None or not np.isscalar(v):
            return "-".rjust(width)
        return (f"{float(v):+.1f}" if sign else f"{float(v):.1f}").rjust(width)

    def _render_panel(frame, positions, mc_row):
        if _panel_text is None:
            return
        g_row = group_per_step[frame] if frame < len(group_per_step) else [0] * num_vehicles
        lines = [_panel_header]
        capped_lines = [""]
        # Every vehicle keeps a row regardless of status so the panel height is stable.
        for i in range(num_vehicles):
            status = _series_val(status_per_step, frame, i) or ""
            completed = status == "completed" or (mc_row is not None and mc_row[i])
            if completed:
                st = "done"
            else:
                st = {"free": "free", "in_corridor": "corr"}.get(
                    status, status[:4] if status else "-")
            cp = str(i)
            live = not completed
            ttr = _series_val(ttr_per_step, frame, i) if live else None
            tgt = _series_val(ttr_target_per_step, frame, i) if live else None
            d = (ttr - tgt) if (ttr is not None and np.isscalar(ttr)
                                and tgt is not None and np.isscalar(tgt)) else None
            nn = None
            if live and g_row[i] >= 0:
                best = np.inf
                for j in range(num_vehicles):
                    if j == i or g_row[j] != g_row[i]:
                        continue
                    best = min(best, float(np.linalg.norm(positions[j] - positions[i])))
                nn = best if np.isfinite(best) else None
            if nn is None:
                nn_str = "-".rjust(6)
            else:
                nn_str = (f"{nn:.2f}*" if nn < separation_distance
                          else f"{nn:.2f} ").rjust(6)
            crit = _series_val(critical_neighbor_per_step, frame, i) if live else None
            crit_h = _series_val(critical_h_per_step, frame, i) if live else None
            if crit is None:
                crit_str = "-".rjust(10)
            elif crit_h is None or not np.isfinite(float(crit_h)):
                crit_str = f"v{int(crit)}".rjust(10)
            else:
                crit_str = f"v{int(crit)}/{float(crit_h):+.2f}".rjust(10)

            prefix = f"{i:>2}  {cp:<5} {st:<4} {_fmt_num(ttr,5)} "
            tgt_str = _fmt_num(tgt, 5)
            lines.append(
                f"{prefix}{tgt_str} {_fmt_num(d,5,sign=True)} {nn_str} {crit_str}")
            capped = live and bool(
                _series_val(ttr_target_capped_per_step, frame, i))
            capped_lines.append(" " * len(prefix) + tgt_str if capped else "")
        _panel_text.set_text("\n".join(lines))
        _panel_capped_text.set_text("\n".join(capped_lines))

    def _render_rich_labels(frame, positions, mc_row):
        for i in range(num_vehicles):
            completed = mc_row is not None and mc_row[i]
            alpha = mission_complete_alpha if completed else 1.0
            xi, yi = positions[i]
            th = float(xs_sim[vehicle_x_dim * i + 2, frame])
            # Flip left-heading labels 180 deg (and mirror the anchor side) so the text stays upright.
            side = 1.0
            th_txt = th
            if np.cos(th) < 0.0:
                side = -1.0
                th_txt = th - np.pi
            lx = xi - side * _label_perp_off * np.sin(th)
            ly = yi + side * _label_perp_off * np.cos(th)
            rich_texts[i].set_position((lx, ly))
            rich_texts[i].set_rotation(np.rad2deg(th_txt))
            rich_texts[i].set_alpha(alpha)
            if verbosity == 0:
                txt = ""
            elif completed:
                txt = "done"
            else:
                parts = []
                rank = _series_val(fs_priority_per_step, frame, i)
                if rank is not None:
                    parts.append(f"P{int(rank)}")
                ttr = _series_val(ttr_per_step, frame, i)
                if ttr is not None and np.isscalar(ttr) and np.isfinite(ttr):
                    parts.append(f"{float(ttr):.1f}")
                txt = "\n".join(parts)
            rich_texts[i].set_text(txt)
        _render_panel(frame, positions, mc_row)

    def init():
        for t in id_texts:
            t.set_text("")
        for t in rich_texts:
            t.set_text("")
        if _panel_text is not None:
            _panel_text.set_text(_panel_header)
            _panel_capped_text.set_text("")
        for sc in trace_scatters:
            sc.set_offsets(np.empty((0, 2)))
        for sc in trace_viol_scatters:
            sc.set_offsets(np.empty((0, 2)))
        for sc in current_scatters:
            sc.set_offsets(np.empty((0, 2)))
        for circle in separation_patches:
            circle.set_center((0.0, 0.0))
            circle.set_facecolor("none")
        for artist in critical_lines + critical_marks:
            artist.set_data([], [])
        if ts_sim is not None and len(ts_sim) > 0:
            title_text.set_text(f"{base_title} - Time: {ts_sim[0]:.2f}")
        artists = trace_scatters + trace_viol_scatters + current_scatters
        artists += critical_lines + critical_marks
        artists.append(title_text)
        artists += separation_patches
        artists += id_texts
        artists += rich_texts
        if _panel_text is not None:
            artists.append(_panel_text)
            artists.append(_panel_capped_text)
        return artists

    def update(frame):
        if ts_sim is not None and frame < len(ts_sim):
            title_text.set_text(f"{base_title} - Time: {ts_sim[frame]:.2f}")

        if priority_list is not None and priority_list[0] is not None:
            i_list = priority_list[frame]
            rank_of_vehicle = [i_list.index(i) + 1 for i in range(num_vehicles)]
        else:
            rank_of_vehicle = [i + 1 for i in range(num_vehicles)]

        positions = np.zeros((num_vehicles, 2), dtype=float)
        n_pts = frame + 1
        for i in range(num_vehicles):
            x_hist = xs_sim[vehicle_x_dim * i, :n_pts]
            y_hist = xs_sim[vehicle_x_dim * i + 1, :n_pts]
            xy = np.column_stack([x_hist, y_hist])
            viol_mask = _violation[i, :n_pts]
            norm_mask = ~viol_mask
            alphas = np.linspace(0.0, trace_alpha, n_pts)

            if np.any(norm_mask):
                trace_scatters[i].set_offsets(xy[norm_mask])
                nc = np.tile(_vehicle_rgba[i], (int(norm_mask.sum()), 1))
                nc[:, 3] = alphas[norm_mask]
                trace_scatters[i].set_facecolors(nc)
            else:
                trace_scatters[i].set_offsets(np.empty((0, 2)))

            if np.any(viol_mask):
                trace_viol_scatters[i].set_offsets(xy[viol_mask])
                vc = np.tile(_red_rgba, (int(viol_mask.sum()), 1))
                vc[:, 3] = np.maximum(alphas[viol_mask], trace_alpha)
                trace_viol_scatters[i].set_facecolors(vc)
            else:
                trace_viol_scatters[i].set_offsets(np.empty((0, 2)))
            if frame >= 0:
                xi = xs_sim[vehicle_x_dim * i, frame]
                yi = xs_sim[vehicle_x_dim * i + 1, frame]
                positions[i, :] = [xi, yi]
                theta_deg = np.rad2deg(xs_sim[vehicle_x_dim * i + 2, frame])
                current_scatters[i].set_offsets([[xi, yi]])
                rot = mtransforms.Affine2D().rotate_deg(theta_deg)
                current_scatters[i].set_paths([_rect_path.transformed(rot)])
                id_texts[i].set_position((xi, yi))
                id_texts[i].set_rotation(theta_deg)
                id_texts[i].set_alpha(1.0)
                id_texts[i].set_text(str(rank_of_vehicle[i]))

        if mission_complete_list is not None and frame < len(mission_complete_list) and mission_complete_list[frame] is not None:
            mc = mission_complete_list[frame]
            for i in range(num_vehicles):
                if mc[i]:
                    current_scatters[i].set_alpha(mission_complete_alpha)
                    trace_scatters[i].set_alpha(mission_complete_alpha)
                    id_texts[i].set_alpha(mission_complete_alpha)
                else:
                    current_scatters[i].set_alpha(1.0)
                    trace_scatters[i].set_alpha(trace_alpha)
                    id_texts[i].set_alpha(1.0)

        _mc = (mission_complete_list[frame]
               if mission_complete_list is not None and frame < len(mission_complete_list)
               else None)
        g_now = group_per_step[frame] if frame < len(group_per_step) else [0] * num_vehicles
        for i in range(num_vehicles):
            xi, yi = positions[i]
            separation_patches[i].set_center((xi, yi))
            if _mc is not None and _mc[i]:
                separation_patches[i].set_facecolor("none")
                separation_patches[i].set_alpha(0.0)
                separation_patches[i].set_edgecolor("none")
                continue
            if np.all(positions == 0.0):
                separation_patches[i].set_facecolor("none")
                separation_patches[i].set_alpha(0.0)
                continue
            deltas = positions - positions[i]
            dists = np.linalg.norm(deltas, axis=1)
            close_mask = (dists > 0.0) & (dists <= separation_distance)
            if g_now[i] < 0:
                close_mask[:] = False
            else:
                for j in range(num_vehicles):
                    if g_now[j] != g_now[i]:
                        close_mask[j] = False
            if np.any(close_mask):
                separation_patches[i].set_facecolor("red")
                separation_patches[i].set_alpha(0.2)
                separation_patches[i].set_edgecolor("none")
            else:
                separation_patches[i].set_facecolor("none")
                separation_patches[i].set_alpha(0.7)
                separation_patches[i].set_edgecolor("k")

        _render_rich_labels(frame, positions, _mc)
        _render_critical_links(frame, positions, _mc)

        artists = trace_scatters + trace_viol_scatters + current_scatters
        artists += critical_lines + critical_marks
        artists.append(title_text)
        artists += separation_patches
        artists += id_texts
        artists += rich_texts
        if _panel_text is not None:
            artists.append(_panel_text)
            artists.append(_panel_capped_text)
        return artists

    # Rotated per-frame labels do not compose with blitting.
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=num_steps, blit=False,
        interval=1000 / fps)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist="ttr_guidance"))
    anim.save(output_path, writer=writer, dpi=150)
    plt.close(fig)
    return anim
