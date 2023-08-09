//! Visualizes multiple vertex placement systems using hand-picked plane QEFs.

use bevy::prelude::*;
use bevy_polyline::{
    prelude::{Polyline, PolylineBundle, PolylineMaterial},
    PolylinePlugin,
};
use nalgebra::{Point3, Rotation3, Vector3};
use qef::Qef3;
use smooth_bevy_cameras::{
    controllers::fps::{FpsCameraBundle, FpsCameraController, FpsCameraPlugin},
    LookTransformPlugin,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(LookTransformPlugin)
        .add_plugin(FpsCameraPlugin::default())
        .add_plugin(PolylinePlugin)
        .add_startup_system(setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut polylines: ResMut<Assets<Polyline>>,
    mut polyline_materials: ResMut<Assets<PolylineMaterial>>,
) {
    let (origins, normals) = two_plane_system(Vector3::zeros());
    let mut qef = Qef3::zeros();
    for (p, n) in origins.into_iter().zip(normals) {
        qef += Qef3::plane(p, n);
        // qef += Qef3::probabilistic_plane(p, n, 0.1, 0.1);
    }
    // let intersection = qef.minimizer_with_exact_inverse();
    let intersection = qef.minimizer_with_pseudo_inverse(0.01);
    render_planes_intersection(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut polylines,
        &mut polyline_materials,
        &origins,
        &normals,
        intersection,
    );

    let (origins, normals) = three_plane_system(Vector3::new(7.0, 0.0, 0.0));
    let mut qef = Qef3::zeros();
    for (p, n) in origins.into_iter().zip(normals) {
        qef += Qef3::plane(p, n);
        // qef += Qef3::probabilistic_plane(p, n, 0.1, 0.1);
    }
    // let intersection = qef.minimizer_with_exact_inverse();
    let intersection = qef.minimizer_with_pseudo_inverse(0.01);
    render_planes_intersection(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut polylines,
        &mut polyline_materials,
        &origins,
        &normals,
        intersection,
    );

    let (origins, normals) = four_plane_system(Vector3::new(14.0, 0.0, 0.0));
    let mut qef = Qef3::zeros();
    for (p, n) in origins.into_iter().zip(normals) {
        qef += Qef3::plane(p, n);
        // qef += Qef3::probabilistic_plane(p, n, 0.1, 0.1);
    }
    // let intersection = qef.minimizer_with_exact_inverse();
    let intersection = qef.minimizer_with_pseudo_inverse(0.01);
    render_planes_intersection(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut polylines,
        &mut polyline_materials,
        &origins,
        &normals,
        intersection,
    );

    commands
        .spawn(Camera3dBundle::default())
        .insert(FpsCameraBundle::new(
            FpsCameraController {
                translate_sensitivity: 3.0,
                ..Default::default()
            },
            Vec3::new(10.0, 10.0, 10.0),
            intersection.into(),
            Vec3::Y,
        ));
}

fn two_plane_system(translate: Vector3<f32>) -> ([Point3<f32>; 2], [Vector3<f32>; 2]) {
    let origins = [
        Point3::new(1.0, 0.0, 0.0) + translate,
        Point3::new(0.0, 1.0, 0.0) + translate,
    ];
    let normals = [
        Vector3::new(1.0, 0.5, 0.0).normalize(),
        Vector3::new(0.0, 1.0, 0.5).normalize(),
    ];
    (origins, normals)
}

fn three_plane_system(translate: Vector3<f32>) -> ([Point3<f32>; 3], [Vector3<f32>; 3]) {
    let origins = [
        Point3::new(1.0, 0.0, 0.0) + translate,
        Point3::new(0.0, 1.0, 0.0) + translate,
        Point3::new(0.0, 0.0, 1.0) + translate,
    ];
    let normals = [
        Vector3::new(1.0, 0.5, 0.0).normalize(),
        Vector3::new(0.0, 1.0, 0.5).normalize(),
        Vector3::new(0.5, 0.0, 1.0).normalize(),
    ];
    (origins, normals)
}

fn four_plane_system(translate: Vector3<f32>) -> ([Point3<f32>; 4], [Vector3<f32>; 4]) {
    let origins = [
        Point3::new(1.0, 0.0, 0.0) + translate,
        Point3::new(0.0, 1.0, 0.0) + translate,
        Point3::new(0.0, 0.0, 1.0) + translate,
        Point3::new(1.0, 2.0, 1.0) + translate,
    ];
    let normals = [
        Vector3::new(1.0, 0.5, 0.0).normalize(),
        Vector3::new(0.0, 1.0, 0.5).normalize(),
        Vector3::new(0.5, 0.0, 1.0).normalize(),
        Vector3::new(-1.0, 2.0, -3.0).normalize(),
    ];
    (origins, normals)
}

fn render_planes_intersection(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    polylines: &mut Assets<Polyline>,
    polyline_materials: &mut Assets<PolylineMaterial>,
    plane_origins: &[Point3<f32>],
    plane_normals: &[Vector3<f32>],
    intersection: Vector3<f32>,
) {
    let plane_mesh = meshes.add(Mesh::from(shape::Plane {
        size: 5.0,
        subdivisions: 1,
    }));
    let sphere_mesh = meshes.add(Mesh::from(shape::UVSphere {
        radius: 0.05,
        sectors: 10,
        stacks: 10,
    }));

    let world_origin_material = materials.add(StandardMaterial {
        unlit: true,
        ..StandardMaterial::from(Color::WHITE)
    });
    let plane_origin_material = materials.add(StandardMaterial {
        unlit: true,
        ..StandardMaterial::from(Color::GREEN)
    });
    let mut plane_color = Color::BLUE;
    plane_color.set_a(0.2);
    let plane_material = materials.add(StandardMaterial {
        unlit: true,
        double_sided: true,
        cull_mode: None,
        ..StandardMaterial::from(plane_color)
    });
    let plane_line_material = polyline_materials.add(PolylineMaterial {
        width: 1.0,
        color: Color::GREEN,
        ..default()
    });

    // Render the world origin.
    commands.spawn(MaterialMeshBundle::<StandardMaterial> {
        mesh: sphere_mesh.clone(),
        material: world_origin_material,
        ..default()
    });

    // Render each plane.
    for (&p, &n) in plane_origins.iter().zip(plane_normals) {
        // Sphere for the plane origin.
        commands.spawn(MaterialMeshBundle::<StandardMaterial> {
            mesh: sphere_mesh.clone(),
            material: plane_origin_material.clone(),
            transform: Transform::from_translation(p.into()),
            ..default()
        });

        // The plane itself.
        let rot = Rotation3::rotation_between(&Vector3::y(), &n).unwrap();
        let transform = Transform {
            translation: p.into(),
            rotation: Quat::from(rot),
            ..default()
        };
        commands.spawn(MaterialMeshBundle::<StandardMaterial> {
            mesh: plane_mesh.clone(),
            material: plane_material.clone(),
            transform,
            ..default()
        });

        // Line between plane origin and intersection point.
        commands.spawn(PolylineBundle {
            polyline: polylines.add(Polyline {
                vertices: vec![p.into(), intersection.into()],
            }),
            material: plane_line_material.clone(),
            ..default()
        });
    }

    // Render the intersection point.
    commands.spawn(MaterialMeshBundle::<StandardMaterial> {
        mesh: sphere_mesh,
        transform: Transform::from_translation(intersection.into()),
        ..default()
    });
}
