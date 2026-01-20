import genesis as gs
from openxr_renderer_fast_genesis import OpenXRRendererFastGenesis


def main():
    gs.init(backend=gs.gpu, precision="32")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.6, 0.0, 0.1)))

    scene.build()
    ctx = scene.visualizer._context

    vr = OpenXRRendererFastGenesis(ctx, resolution_scale=1.0)
    vr.build()

    try:
        while not vr.should_quit:
            vr.poll_events()
            scene.step()
            vr.update_scene(force_render=True)
            vr.render_frame()
    finally:
        vr.destroy()


if __name__ == "__main__":
    main()
